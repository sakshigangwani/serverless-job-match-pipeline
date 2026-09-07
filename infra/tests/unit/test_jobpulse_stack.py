import aws_cdk as core
import aws_cdk.assertions as assertions

from jobpulse_infra.jobpulse_stack import JobPulseStack


def _synth_template() -> assertions.Template:
    app = core.App()
    stack = JobPulseStack(app, "jobpulse-test")
    return assertions.Template.from_stack(stack)


def test_raw_postings_bucket_blocks_public_access():
    template = _synth_template()

    template.has_resource_properties(
        "AWS::S3::Bucket",
        {
            "PublicAccessBlockConfiguration": {
                "BlockPublicAcls": True,
                "BlockPublicPolicy": True,
                "IgnorePublicAcls": True,
                "RestrictPublicBuckets": True,
            }
        },
    )


def test_common_layer_targets_python312():
    template = _synth_template()

    template.has_resource_properties(
        "AWS::Lambda::LayerVersion",
        {"CompatibleRuntimes": ["python3.12"]},
    )


def test_fetch_lambda_uses_python312_and_common_layer():
    template = _synth_template()

    template.has_resource_properties(
        "AWS::Lambda::Function",
        {
            "Handler": "handler.handler",
            "Runtime": "python3.12",
            "Environment": {
                "Variables": {
                    "JOB_SOURCE": "remoteok",
                }
            },
        },
    )
    template.resource_count_is("AWS::Lambda::LayerVersion", 1)


def test_fetch_schedule_rule_runs_every_6_hours():
    template = _synth_template()

    template.has_resource_properties(
        "AWS::Events::Rule",
        {"ScheduleExpression": "rate(6 hours)", "State": "ENABLED"},
    )


def test_fetch_lambda_role_only_grants_s3_write_not_read():
    template = _synth_template()

    template.has_resource_properties(
        "AWS::IAM::Policy",
        {
            "PolicyDocument": {
                "Statement": assertions.Match.array_with(
                    [
                        assertions.Match.object_like(
                            {
                                "Action": assertions.Match.array_with(["s3:PutObject"]),
                                "Effect": "Allow",
                            }
                        )
                    ]
                )
            }
        },
    )
