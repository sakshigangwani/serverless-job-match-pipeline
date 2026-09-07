import aws_cdk as core
from aws_cdk import assertions
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
    template.resource_count_is("AWS::Lambda::LayerVersion", 2)


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


def test_pydantic_layer_targets_x86_64():
    template = _synth_template()

    template.has_resource_properties(
        "AWS::Lambda::LayerVersion",
        {
            "CompatibleRuntimes": ["python3.12"],
            "CompatibleArchitectures": ["x86_64"],
        },
    )


def test_extract_lambda_has_both_layers_and_bedrock_model_env():
    template = _synth_template()

    matches = template.find_resources(
        "AWS::Lambda::Function",
        {
            "Properties": {
                "Handler": "handler.handler",
                "Runtime": "python3.12",
                "Environment": {
                    "Variables": {"BEDROCK_MODEL_ID": assertions.Match.any_value()}
                },
            }
        },
    )

    assert len(matches) == 1
    (resource,) = matches.values()
    assert len(resource["Properties"]["Layers"]) == 2


def test_extract_lambda_role_reads_raw_and_only_puts_new_dynamodb_items():
    template = _synth_template()

    template.has_resource_properties(
        "AWS::IAM::Policy",
        {
            "PolicyDocument": {
                "Statement": assertions.Match.array_with(
                    [
                        assertions.Match.object_like(
                            {"Action": "s3:GetObject", "Effect": "Allow"}
                        ),
                        assertions.Match.object_like(
                            {"Action": "dynamodb:PutItem", "Effect": "Allow"}
                        ),
                    ]
                )
            }
        },
    )


def test_extract_lambda_role_can_invoke_only_anthropic_bedrock_models():
    template = _synth_template()

    template.has_resource_properties(
        "AWS::IAM::Policy",
        {
            "PolicyDocument": {
                "Statement": assertions.Match.array_with(
                    [
                        assertions.Match.object_like(
                            {
                                "Action": "bedrock:InvokeModel",
                                "Effect": "Allow",
                                "Resource": assertions.Match.object_like(
                                    {
                                        "Fn::Join": assertions.Match.array_with(
                                            [
                                                assertions.Match.array_with(
                                                    [
                                                        "::foundation-model/anthropic.*"
                                                    ]
                                                )
                                            ]
                                        )
                                    }
                                ),
                            }
                        )
                    ]
                )
            }
        },
    )


def test_raw_postings_bucket_notifies_extract_lambda_on_new_raw_objects():
    template = _synth_template()

    template.has_resource_properties(
        "Custom::S3BucketNotifications",
        {
            "NotificationConfiguration": {
                "LambdaFunctionConfigurations": assertions.Match.array_with(
                    [
                        assertions.Match.object_like(
                            {
                                "Filter": {
                                    "Key": {
                                        "FilterRules": assertions.Match.array_with(
                                            [
                                                {"Name": "suffix", "Value": ".json"},
                                                {"Name": "prefix", "Value": "raw/"},
                                            ]
                                        )
                                    }
                                }
                            }
                        )
                    ]
                )
            }
        },
    )


def test_embed_lambda_has_both_layers_and_embedding_model_env():
    template = _synth_template()

    matches = template.find_resources(
        "AWS::Lambda::Function",
        {
            "Properties": {
                "Handler": "handler.handler",
                "Runtime": "python3.12",
                "Environment": {
                    "Variables": {
                        "BEDROCK_EMBEDDING_MODEL_ID": assertions.Match.any_value()
                    }
                },
            }
        },
    )

    assert len(matches) == 1
    (resource,) = matches.values()
    assert len(resource["Properties"]["Layers"]) == 2


def test_embed_lambda_role_scoped_to_candidate_prefix_and_only_updates_dynamodb():
    template = _synth_template()

    template.has_resource_properties(
        "AWS::IAM::Policy",
        {
            "PolicyDocument": {
                "Statement": assertions.Match.array_with(
                    [
                        assertions.Match.object_like(
                            {
                                "Action": ["s3:GetObject", "s3:PutObject"],
                                "Effect": "Allow",
                                "Resource": assertions.Match.object_like(
                                    {
                                        "Fn::Join": assertions.Match.array_with(
                                            [
                                                assertions.Match.array_with(
                                                    ["/candidate/*"]
                                                )
                                            ]
                                        )
                                    }
                                ),
                            }
                        ),
                        assertions.Match.object_like(
                            {"Action": "dynamodb:UpdateItem", "Effect": "Allow"}
                        ),
                    ]
                )
            }
        },
    )


def test_embed_lambda_role_can_invoke_only_titan_embedding_models():
    template = _synth_template()

    template.has_resource_properties(
        "AWS::IAM::Policy",
        {
            "PolicyDocument": {
                "Statement": assertions.Match.array_with(
                    [
                        assertions.Match.object_like(
                            {
                                "Action": "bedrock:InvokeModel",
                                "Effect": "Allow",
                                "Resource": assertions.Match.object_like(
                                    {
                                        "Fn::Join": assertions.Match.array_with(
                                            [
                                                assertions.Match.array_with(
                                                    [
                                                        "::foundation-model/amazon.titan-embed*"
                                                    ]
                                                )
                                            ]
                                        )
                                    }
                                ),
                            }
                        )
                    ]
                )
            }
        },
    )


def test_postings_table_has_score_ranked_gsi_and_streams_enabled():
    template = _synth_template()

    template.has_resource_properties(
        "AWS::DynamoDB::Table",
        {
            "BillingMode": "PAY_PER_REQUEST",
            "KeySchema": [{"AttributeName": "posting_id", "KeyType": "HASH"}],
            "StreamSpecification": {"StreamViewType": "NEW_IMAGE"},
            "GlobalSecondaryIndexes": assertions.Match.array_with(
                [
                    assertions.Match.object_like(
                        {
                            "IndexName": "ScoreIndex",
                            "KeySchema": [
                                {"AttributeName": "gsi_pk", "KeyType": "HASH"},
                                {"AttributeName": "score", "KeyType": "RANGE"},
                            ],
                        }
                    )
                ]
            ),
        },
    )


def test_embed_lambda_dynamodb_stream_trigger_is_filtered_to_insert_only():
    template = _synth_template()

    template.has_resource_properties(
        "AWS::Lambda::EventSourceMapping",
        {
            "StartingPosition": "LATEST",
            "FilterCriteria": {
                "Filters": [{"Pattern": '{"eventName":["INSERT"]}'}]
            },
        },
    )


def test_only_the_raw_prefix_notification_remains_on_the_bucket():
    template = _synth_template()

    matches = template.find_resources("Custom::S3BucketNotifications")
    (notification,) = matches.values()
    configs = notification["Properties"]["NotificationConfiguration"][
        "LambdaFunctionConfigurations"
    ]

    assert len(configs) == 1
    prefixes = {
        rule["Value"]
        for rule in configs[0]["Filter"]["Key"]["FilterRules"]
        if rule["Name"] == "prefix"
    }
    assert prefixes == {"raw/"}


def test_threshold_lambda_env_and_layers():
    template = _synth_template()

    # ALERTS_TOPIC_ARN, not FIT_THRESHOLD, disambiguates threshold_lambda: skill_gap_lambda
    # (Phase 14) also reads FIT_THRESHOLD, but only threshold_lambda publishes alerts.
    matches = template.find_resources(
        "AWS::Lambda::Function",
        {
            "Properties": {
                "Handler": "handler.handler",
                "Runtime": "python3.12",
                "Environment": {
                    "Variables": {"ALERTS_TOPIC_ARN": assertions.Match.any_value()}
                },
            }
        },
    )

    assert len(matches) == 1
    (resource,) = matches.values()
    assert len(resource["Properties"]["Layers"]) == 2


def test_threshold_lambda_role_only_updates_dynamodb_and_publishes_to_sns():
    template = _synth_template()

    template.has_resource_properties(
        "AWS::IAM::Policy",
        {
            "PolicyDocument": {
                "Statement": assertions.Match.array_with(
                    [
                        assertions.Match.object_like(
                            {"Action": "dynamodb:UpdateItem", "Effect": "Allow"}
                        ),
                        assertions.Match.object_like(
                            {"Action": "sns:Publish", "Effect": "Allow"}
                        ),
                    ]
                )
            }
        },
    )


def test_threshold_dynamodb_trigger_fires_only_after_embedding_before_scoring():
    template = _synth_template()

    template.has_resource_properties(
        "AWS::Lambda::EventSourceMapping",
        {
            "StartingPosition": "LATEST",
            "FilterCriteria": {
                "Filters": [
                    {
                        "Pattern": (
                            '{"eventName":["MODIFY"],"dynamodb":{"NewImage":'
                            '{"embedding":{"B":[{"exists":true}]},'
                            '"score":{"NULL":[true]}}}}'
                        )
                    }
                ]
            },
        },
    )


def test_alerts_topic_exists_and_threshold_can_publish_to_it():
    template = _synth_template()

    template.resource_count_is("AWS::SNS::Topic", 1)


def test_alert_email_lambda_can_only_send_ses():
    # Layer count (CommonLayer only, for logging/metrics — PLAN.md Phase 9) is covered
    # by test_alert_email_lambda_has_common_layer_for_logging_and_metrics.
    template = _synth_template()

    template.has_resource_properties(
        "AWS::IAM::Policy",
        {
            "PolicyDocument": {
                "Statement": assertions.Match.array_with(
                    [
                        assertions.Match.object_like(
                            {
                                "Action": ["ses:SendEmail", "ses:SendRawEmail"],
                                "Effect": "Allow",
                            }
                        )
                    ]
                )
            }
        },
    )


def test_alert_email_lambda_is_subscribed_to_the_alerts_topic():
    template = _synth_template()

    template.resource_count_is("AWS::SNS::Subscription", 1)
    template.has_resource_properties("AWS::SNS::Subscription", {"Protocol": "lambda"})


def test_query_api_lambda_env_and_layers():
    template = _synth_template()

    matches = template.find_resources(
        "AWS::Lambda::Function",
        {
            "Properties": {
                "Handler": "handler.handler",
                "Environment": {
                    "Variables": assertions.Match.exact(
                        {"POSTINGS_TABLE_NAME": assertions.Match.any_value()}
                    )
                },
            }
        },
    )
    assert len(matches) == 1
    (resource,) = matches.values()
    assert len(resource["Properties"]["Layers"]) == 2


def test_query_api_lambda_role_only_queries_table_and_gsi():
    template = _synth_template()

    template.has_resource_properties(
        "AWS::IAM::Policy",
        {
            "PolicyDocument": {
                "Statement": assertions.Match.array_with(
                    [
                        assertions.Match.object_like(
                            {
                                "Action": "dynamodb:Query",
                                "Effect": "Allow",
                                "Resource": assertions.Match.array_with(
                                    [
                                        assertions.Match.object_like(
                                            {
                                                "Fn::Join": assertions.Match.array_with(
                                                    [
                                                        assertions.Match.array_with(
                                                            ["/index/*"]
                                                        )
                                                    ]
                                                )
                                            }
                                        )
                                    ]
                                ),
                            }
                        )
                    ]
                )
            }
        },
    )


def test_query_api_requires_an_api_key():
    template = _synth_template()

    template.resource_count_is("AWS::ApiGateway::ApiKey", 1)
    template.resource_count_is("AWS::ApiGateway::UsagePlan", 1)
    template.resource_count_is("AWS::ApiGateway::UsagePlanKey", 1)

    matches = template.find_resources("AWS::ApiGateway::Method", {"Properties": {"HttpMethod": "ANY"}})
    assert len(matches) >= 1
    assert all(m["Properties"]["ApiKeyRequired"] is True for m in matches.values())


def test_skill_gap_lambda_env_and_layers():
    template = _synth_template()

    matches = template.find_resources(
        "AWS::Lambda::Function",
        {
            "Properties": {
                "Handler": "handler.handler",
                "Environment": {
                    "Variables": assertions.Match.exact(
                        {
                            "POSTINGS_TABLE_NAME": assertions.Match.any_value(),
                            "FIT_THRESHOLD": assertions.Match.any_value(),
                        }
                    )
                },
            }
        },
    )
    assert len(matches) == 1
    (resource,) = matches.values()
    assert len(resource["Properties"]["Layers"]) == 2


def test_skill_gap_lambda_role_only_queries_table_and_gsi():
    template = _synth_template()

    matches = template.find_resources(
        "AWS::IAM::Policy",
        {
            "Properties": {
                "PolicyDocument": {
                    "Statement": assertions.Match.array_with(
                        [
                            assertions.Match.object_like(
                                {"Action": "dynamodb:Query", "Effect": "Allow"}
                            )
                        ]
                    )
                }
            }
        },
    )
    # query_api_lambda and skill_gap_lambda each get their own Query-only policy.
    assert len(matches) == 2


def test_skill_gap_endpoint_exists_on_the_query_api_and_requires_a_key():
    template = _synth_template()

    template.has_resource_properties("AWS::ApiGateway::Resource", {"PathPart": "skill-gap"})

    # query_api's own routes (root + {proxy+}) are both "ANY"; GET is unique to the
    # skill-gap resource's method.
    matches = template.find_resources("AWS::ApiGateway::Method", {"Properties": {"HttpMethod": "GET"}})
    assert len(matches) == 1
    (resource,) = matches.values()
    assert resource["Properties"]["ApiKeyRequired"] is True

    # Still exactly one API Gateway REST API, one API key, one usage plan — skill-gap
    # is a sibling resource on the same API, not a second API.
    template.resource_count_is("AWS::ApiGateway::RestApi", 1)
    template.resource_count_is("AWS::ApiGateway::ApiKey", 1)
    template.resource_count_is("AWS::ApiGateway::UsagePlan", 1)


def test_every_pipeline_lambda_has_active_xray_tracing():
    template = _synth_template()

    matches = template.find_resources(
        "AWS::Lambda::Function", {"Properties": {"TracingConfig": {"Mode": "Active"}}}
    )
    # fetch, extract, embed, threshold, alert_email, query_api, dlq_redrive, skill_gap —
    # not the two CDK-provided custom-resource handlers (S3 auto-delete, bucket
    # notifications), which CDK doesn't enable tracing on.
    assert len(matches) == 8


def test_query_api_stage_has_xray_tracing_enabled():
    template = _synth_template()

    template.has_resource_properties("AWS::ApiGateway::Stage", {"TracingEnabled": True})


def test_alert_email_lambda_has_common_layer_for_logging_and_metrics():
    template = _synth_template()

    matches = template.find_resources(
        "AWS::Lambda::Function",
        {
            "Properties": {
                "Environment": {
                    "Variables": assertions.Match.object_like(
                        {"SENDER_EMAIL": assertions.Match.any_value()}
                    )
                }
            }
        },
    )
    assert len(matches) == 1
    (resource,) = matches.values()
    assert len(resource["Properties"]["Layers"]) == 1


def test_every_pipeline_lambda_has_an_error_rate_and_duration_alarm():
    template = _synth_template()

    # 7 lambdas x 2 alarms each, plus 2 DLQ depth alarms (PLAN.md Phase 12.3).
    template.resource_count_is("AWS::CloudWatch::Alarm", 16)

    error_rate_alarms = template.find_resources(
        "AWS::CloudWatch::Alarm",
        {
            "Properties": {
                "Metrics": assertions.Match.array_with(
                    [
                        assertions.Match.object_like(
                            {"Expression": "(errors / invocations) * 100"}
                        )
                    ]
                )
            }
        },
    )
    assert len(error_rate_alarms) == 7


def test_error_rate_alarms_do_not_false_alarm_on_an_idle_lambda():
    template = _synth_template()

    template.has_resource_properties(
        "AWS::CloudWatch::Alarm",
        {
            "Metrics": assertions.Match.array_with(
                [assertions.Match.object_like({"Expression": "(errors / invocations) * 100"})]
            ),
            "TreatMissingData": "notBreaching",
        },
    )


def test_pipeline_dashboard_exists():
    template = _synth_template()

    template.resource_count_is("AWS::CloudWatch::Dashboard", 1)
    template.has_resource_properties(
        "AWS::CloudWatch::Dashboard", {"DashboardName": "JobPulse-Pipeline"}
    )


def test_extract_lambda_has_an_async_dlq_with_retry_and_max_event_age():
    template = _synth_template()

    template.resource_count_is("AWS::SQS::Queue", 2)
    template.has_resource_properties(
        "AWS::Lambda::EventInvokeConfig",
        {
            "MaximumRetryAttempts": 2,
            "MaximumEventAgeInSeconds": 7200,
            "DestinationConfig": {"OnFailure": {"Destination": assertions.Match.any_value()}},
        },
    )


def test_embed_event_source_mapping_has_a_dlq_destination():
    template = _synth_template()

    template.has_resource_properties(
        "AWS::Lambda::EventSourceMapping",
        {
            "DestinationConfig": {"OnFailure": {"Destination": assertions.Match.any_value()}},
            # Still the same INSERT-only filter from Phase 5 — a DLQ destination
            # doesn't change what triggers this Lambda, only what happens on failure.
            "FilterCriteria": {"Filters": [{"Pattern": '{"eventName":["INSERT"]}'}]},
        },
    )


def test_dlq_redrive_lambda_can_consume_both_dlqs_invoke_both_lambdas_and_scan_the_table():
    template = _synth_template()

    template.has_resource_properties(
        "AWS::IAM::Policy",
        {
            "PolicyDocument": {
                "Statement": assertions.Match.array_with(
                    [
                        assertions.Match.object_like(
                            {"Action": assertions.Match.array_with(["sqs:DeleteMessage"])}
                        ),
                        assertions.Match.object_like({"Action": "lambda:InvokeFunction"}),
                        assertions.Match.object_like(
                            {"Action": "dynamodb:Scan", "Effect": "Allow"}
                        ),
                    ]
                )
            }
        },
    )


def test_dlq_depth_alarms_fire_on_any_message_not_a_percentage():
    template = _synth_template()

    matches = template.find_resources(
        "AWS::CloudWatch::Alarm",
        {
            "Properties": {
                "MetricName": "ApproximateNumberOfMessagesVisible",
                "Threshold": 1,
                "TreatMissingData": "notBreaching",
            }
        },
    )
    assert len(matches) == 2
