import re
import shutil
import subprocess
import sys
from pathlib import Path

from aws_cdk import (
    Duration,
    RemovalPolicy,
    Stack,
)
from aws_cdk import (
    aws_events as events,
)
from aws_cdk import (
    aws_events_targets as targets,
)
from aws_cdk import (
    aws_iam as iam,
)
from aws_cdk import (
    aws_lambda as _lambda,
)
from aws_cdk import (
    aws_s3 as s3,
)
from aws_cdk import (
    aws_s3_notifications as s3n,
)
from constructs import Construct

REPO_ROOT = Path(__file__).resolve().parents[2]
LAMBDAS_DIR = REPO_ROOT / "lambdas"
LAYER_BUILD_ROOT = Path(__file__).resolve().parent / ".layer_build"

DEFAULT_BEDROCK_MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"


def _pinned_version(package: str) -> str:
    """Read a package's pinned version out of the root requirements.txt, so the Lambda
    layer we build for it can never silently drift from what's pinned for local dev.
    """
    requirements = (REPO_ROOT / "requirements.txt").read_text()
    match = re.search(rf"^{re.escape(package)}==([^\s;]+)", requirements, re.MULTILINE)
    if not match:
        raise ValueError(f"{package} is not pinned in requirements.txt")
    return match.group(1)


def _build_common_layer_asset() -> str:
    """Copy common/*.py (+ example JSON) into a Lambda-layer-shaped python/common/ dir.

    A plain file copy, not pip/Docker bundling: common/storage_keys.py has no
    third-party dependency, so it's safe to ship as-is. common/models.py imports
    pydantic, but that import is only exercised by whichever Lambda actually does
    `from common.models import ...` (common/__init__.py has no eager imports) — a Lambda
    that needs it also attaches the PydanticLayer built below.
    """
    build_dir = LAYER_BUILD_ROOT / "common"
    python_dir = build_dir / "python" / "common"
    if python_dir.exists():
        shutil.rmtree(python_dir)
    python_dir.mkdir(parents=True)
    for pattern in ("*.py", "*.json"):
        for file_path in (REPO_ROOT / "common").glob(pattern):
            shutil.copy(file_path, python_dir / file_path.name)
    return str(build_dir)


def _build_pydantic_layer_asset() -> str:
    """Download pydantic (+ its compiled pydantic-core dependency) as Lambda-compatible
    manylinux wheels, without Docker.

    pydantic-core ships as a platform-specific compiled wheel (it's written in Rust).
    Building it locally with a plain `pip install --target` on a Mac would silently
    bundle a macOS binary that crashes on Lambda's Linux runtime. Instead, `pip install
    --platform manylinux2014_x86_64 --only-binary=:all:` downloads the prebuilt Linux
    wheel straight from PyPI — no compilation, no Docker — as long as one exists for the
    pinned version (it does for pydantic 2.x). Cached by pinned version so `cdk synth`
    doesn't re-download on every run.
    """
    version = _pinned_version("pydantic")
    build_dir = LAYER_BUILD_ROOT / "pydantic"
    marker = build_dir / f".built-{version}"
    if marker.exists():
        return str(build_dir)

    python_dir = build_dir / "python"
    if python_dir.exists():
        shutil.rmtree(python_dir)
    python_dir.mkdir(parents=True)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            f"pydantic=={version}",
            "--platform",
            "manylinux2014_x86_64",
            "--implementation",
            "cp",
            "--python-version",
            "3.12",
            "--only-binary=:all:",
            "--target",
            str(python_dir),
        ],
        check=True,
        capture_output=True,
    )
    marker.touch()
    return str(build_dir)


class JobPulseStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # --- Raw postings landing zone (PLAN.md Phase 2.2) ---
        # Immutable audit trail: every posting a source returns lands here untouched,
        # keyed by common.storage_keys.raw_posting_key. Also holds the extraction
        # Lambda's structured/ output (PLAN.md Phase 3) until Phase 5 adds DynamoDB.
        raw_postings_bucket = s3.Bucket(
            self,
            "RawPostingsBucket",
            removal_policy=RemovalPolicy.DESTROY,  # dev default; use RETAIN in prod
            auto_delete_objects=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
        )

        # --- Shared data-contracts layer, consumed by every Lambda (Phase 1 code) ---
        common_layer = _lambda.LayerVersion(
            self,
            "CommonLayer",
            code=_lambda.Code.from_asset(_build_common_layer_asset()),
            compatible_runtimes=[_lambda.Runtime.PYTHON_3_12],
            description="Shared JobPulse data contracts (common/ package)",
        )

        # --- Third-party dependency layer, for Lambdas that need common.models ---
        pydantic_layer = _lambda.LayerVersion(
            self,
            "PydanticLayer",
            code=_lambda.Code.from_asset(_build_pydantic_layer_asset()),
            compatible_runtimes=[_lambda.Runtime.PYTHON_3_12],
            compatible_architectures=[_lambda.Architecture.X86_64],
            description="pydantic + pydantic-core (manylinux x86_64 wheels)",
        )

        # --- Ingestion Lambda (PLAN.md Phase 2.1) ---
        fetch_lambda = _lambda.Function(
            self,
            "FetchLambda",
            runtime=_lambda.Runtime.PYTHON_3_12,
            architecture=_lambda.Architecture.X86_64,
            handler="handler.handler",
            code=_lambda.Code.from_asset(str(LAMBDAS_DIR / "fetch")),
            layers=[common_layer],
            timeout=Duration.seconds(30),
            memory_size=256,
            environment={
                "RAW_BUCKET_NAME": raw_postings_bucket.bucket_name,
                "JOB_SOURCE": "remoteok",
            },
        )
        raw_postings_bucket.grant_write(fetch_lambda)

        # --- Scheduled trigger (PLAN.md Phase 2.3) ---
        events.Rule(
            self,
            "FetchScheduleRule",
            schedule=events.Schedule.rate(Duration.hours(6)),
            targets=[targets.LambdaFunction(fetch_lambda)],
        )

        # --- Extraction Lambda (PLAN.md Phase 3) ---
        extract_lambda = _lambda.Function(
            self,
            "ExtractLambda",
            runtime=_lambda.Runtime.PYTHON_3_12,
            architecture=_lambda.Architecture.X86_64,
            handler="handler.handler",
            code=_lambda.Code.from_asset(str(LAMBDAS_DIR / "extract")),
            layers=[common_layer, pydantic_layer],
            timeout=Duration.seconds(60),  # allows for the Bedrock call + one retry
            memory_size=512,
            environment={"BEDROCK_MODEL_ID": DEFAULT_BEDROCK_MODEL_ID},
        )
        # Least privilege: read only the raw/ prefix it consumes, write only the
        # structured/ prefix it produces — not full read/write on the whole bucket.
        extract_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject"],
                resources=[raw_postings_bucket.arn_for_objects("raw/*")],
            )
        )
        extract_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:PutObject"],
                resources=[raw_postings_bucket.arn_for_objects("structured/*")],
            )
        )
        extract_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel"],
                resources=[
                    f"arn:aws:bedrock:{self.region}::foundation-model/anthropic.*"
                ],
            )
        )

        # --- Event-driven trigger: new raw posting -> extract (PLAN.md Phase 3) ---
        raw_postings_bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3n.LambdaDestination(extract_lambda),
            s3.NotificationKeyFilter(prefix="raw/", suffix=".json"),
        )

        self.raw_postings_bucket = raw_postings_bucket
        self.common_layer = common_layer
        self.pydantic_layer = pydantic_layer
        self.fetch_lambda = fetch_lambda
        self.extract_lambda = extract_lambda
