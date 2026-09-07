import os

import aws_cdk as cdk
from jobpulse_infra.jobpulse_stack import JobPulseStack

app = cdk.App()
JobPulseStack(
    app,
    "JobPulseStack",
    # Bound to whatever account/region the CDK CLI's current credentials resolve to
    # (CDK populates CDK_DEFAULT_ACCOUNT/CDK_DEFAULT_REGION from the active AWS
    # session at synth time) rather than a hardcoded account. This is what makes the
    # Phase 10 CI/CD staging/production split work: each GitHub Environment assumes a
    # different AWS account's OIDC role, so the same `cdk deploy` command deploys to a
    # different account depending on which environment's job is running.
    env=cdk.Environment(
        account=os.getenv("CDK_DEFAULT_ACCOUNT"), region=os.getenv("CDK_DEFAULT_REGION")
    ),
)

app.synth()
