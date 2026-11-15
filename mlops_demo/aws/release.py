import json
from hashlib import sha256

from botocore.client import BaseClient
from botocore.waiter import WaiterModel, create_waiter_with_client


def deploy_and_validate(
    client: BaseClient,
    *,
    pipeline_name: str,
    pipeline_role: str,
    definition: dict,
    staging_source: str,
    staging_group: str,
) -> dict[str, str | int]:
    document = json.dumps(definition, separators=(",", ":"), sort_keys=True)
    request = {
        "PipelineName": pipeline_name,
        "RoleArn": pipeline_role,
        "PipelineDefinition": document,
    }
    try:
        client.describe_pipeline(PipelineName=pipeline_name)
    except client.exceptions.ResourceNotFound:
        deployed = client.create_pipeline(
            **request,
            ClientRequestToken=_token([pipeline_name, pipeline_role, document]),
        )
        versions = client.list_pipeline_versions(
            PipelineName=pipeline_name, MaxResults=1, SortOrder="Descending"
        )["PipelineVersionSummaries"]
        if not versions:
            raise RuntimeError("SageMaker has not returned the first pipeline version.")
        version = versions[0]["PipelineVersionId"]
        recorded = client.describe_pipeline(PipelineName=pipeline_name, PipelineVersionId=version)
        if json.loads(recorded["PipelineDefinition"]) != definition:
            raise RuntimeError("The first SageMaker pipeline version does not match the deployed definition.")
    else:
        deployed = client.update_pipeline(**request)
        version = deployed["PipelineVersionId"]

    parameters = [
        {"Name": "SourceUri", "Value": staging_source},
        {"Name": "ModelGroup", "Value": staging_group},
        {"Name": "RequireApproval", "Value": "true"},
    ]
    execution = client.start_pipeline_execution(
        PipelineName=pipeline_name,
        PipelineVersionId=version,
        PipelineParameters=parameters,
        ClientRequestToken=_token([pipeline_name, version, parameters]),
    )["PipelineExecutionArn"]
    _wait_for_validation(client, execution)
    return {
        "pipeline_name": pipeline_name,
        "pipeline_arn": deployed["PipelineArn"],
        "pipeline_version": version,
        "staging_execution_arn": execution,
    }


def schedule_validated_version(
    client: BaseClient,
    *,
    name: str,
    expression: str,
    timezone: str,
    scheduler_role: str,
    release: dict[str, str | int],
) -> None:
    partition = str(release["pipeline_arn"]).split(":")[1]
    payload = {
        "PipelineName": release["pipeline_name"],
        "PipelineVersionId": release["pipeline_version"],
        "ClientRequestToken": "scheduled-retraining-<aws.scheduler.scheduled-time>",
    }
    request = {
        "Name": name,
        "ScheduleExpression": expression,
        "ScheduleExpressionTimezone": timezone,
        "State": "ENABLED",
        "FlexibleTimeWindow": {"Mode": "OFF"},
        "Target": {
            "Arn": f"arn:{partition}:scheduler:::aws-sdk:sagemaker:startPipelineExecution",
            "RoleArn": scheduler_role,
            "Input": json.dumps(payload),
            "RetryPolicy": {"MaximumRetryAttempts": 2, "MaximumEventAgeInSeconds": 300},
        },
    }
    try:
        current = client.get_schedule(Name=name)
    except client.exceptions.ResourceNotFoundException:
        client.create_schedule(**request, ClientToken=_token(["create", request]))
    else:
        pinned = json.loads(current["Target"]["Input"])
        if pinned["PipelineName"] != release["pipeline_name"]:
            raise ValueError("The existing retraining schedule belongs to a different pipeline.")
        if pinned["PipelineVersionId"] > release["pipeline_version"]:
            raise ValueError("The existing retraining schedule already points to a newer validated version.")
        client.update_schedule(**request, ClientToken=_token(["update", request]))


def _wait_for_validation(client: BaseClient, execution: str) -> None:
    model = WaiterModel({
        "version": 2,
        "waiters": {
            "PipelineValidated": {
                "operation": "DescribePipelineExecution",
                "delay": 15,
                "maxAttempts": 120,
                "acceptors": [
                    {"matcher": "path", "argument": "PipelineExecutionStatus", "expected": status, "state": state}
                    for status, state in (
                        ("Succeeded", "success"),
                        ("Failed", "failure"),
                        ("Stopped", "failure"),
                    )
                ],
            },
        },
    })
    create_waiter_with_client("PipelineValidated", model, client).wait(PipelineExecutionArn=execution)


def _token(values: list) -> str:
    return sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
