import shutil
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
    aws_lambda as _lambda,
)
from aws_cdk import (
    aws_s3 as s3,
)
from constructs import Construct

REPO_ROOT = Path(__file__).resolve().parents[2]
LAMBDAS_DIR = REPO_ROOT / "lambdas"
LAYER_BUILD_DIR = Path(__file__).resolve().parent / ".layer_build" / "common"


def _build_common_layer_asset() -> str:
    """Copy common/*.py (+ example JSON) into a Lambda-layer-shaped python/common/ dir.

    A plain file copy, not pip/Docker bundling: common/storage_keys.py has no
    third-party dependency, so it's safe to ship as-is. common/models.py imports
    pydantic, but that import is only exercised by whichever Lambda actually does
    `from common.models import ...` (common/__init__.py has no eager imports) — that
    Lambda is responsible for bundling pydantic itself when it's wired up in a later
    phase. Phase 2's fetch Lambda only needs common.storage_keys.
    """
    common_src = REPO_ROOT / "common"
    python_dir = LAYER_BUILD_DIR / "python" / "common"
    if python_dir.exists():
        shutil.rmtree(python_dir)
    python_dir.mkdir(parents=True)
    for pattern in ("*.py", "*.json"):
        for file_path in common_src.glob(pattern):
            shutil.copy(file_path, python_dir / file_path.name)
    return str(LAYER_BUILD_DIR)


class JobPulseStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # --- Raw postings landing zone (PLAN.md Phase 2.2) ---
        # Immutable audit trail: every posting a source returns lands here untouched,
        # keyed by common.storage_keys.raw_posting_key.
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

        # --- Ingestion Lambda (PLAN.md Phase 2.1) ---
        fetch_lambda = _lambda.Function(
            self,
            "FetchLambda",
            runtime=_lambda.Runtime.PYTHON_3_12,
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

        self.raw_postings_bucket = raw_postings_bucket
        self.common_layer = common_layer
        self.fetch_lambda = fetch_lambda
