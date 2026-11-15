import re

from .processing import split_s3_uri


def build_pipeline_definition(
    *,
    image_uri: str,
    processing_role: str,
    source_uri: str,
    artifact_root: str,
    model_group: str,
    instance_type: str = "ml.m5.large",
) -> dict:
    ecr_digest = r"[0-9]{12}\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com(?:\.cn)?/[^:@\s]+@sha256:[a-f0-9]{64}"
    if not re.fullmatch(ecr_digest, image_uri):
        raise ValueError("Use an ECR image URI pinned by SHA-256 digest.")
    split_s3_uri(source_uri)
    artifact_root = artifact_root.rstrip("/")
    split_s3_uri(artifact_root)

    parameters = [
        {"Name": "SourceUri", "Type": "String", "DefaultValue": source_uri},
        {"Name": "ModelGroup", "Type": "String", "DefaultValue": model_group},
        {"Name": "Seed", "Type": "String", "DefaultValue": "42"},
        {"Name": "MaxMae", "Type": "String", "DefaultValue": "5"},
        {"Name": "RequireApproval", "Type": "Boolean", "DefaultValue": False},
    ]
    execution = {"Get": "Execution.PipelineExecutionId"}
    processing = {
        "Name": "TrainEvaluate",
        "Type": "Processing",
        "Arguments": {
            "RoleArn": processing_role,
            "AppSpecification": {
                "ImageUri": image_uri,
                "ContainerArguments": [
                    "process",
                    "--source-uri", {"Get": "Parameters.SourceUri"},
                    "--seed", {"Get": "Parameters.Seed"},
                    "--max-mae", {"Get": "Parameters.MaxMae"},
                ],
            },
            "ProcessingResources": {
                "ClusterConfig": {
                    "InstanceCount": 1,
                    "InstanceType": instance_type,
                    "VolumeSizeInGB": 10,
                },
            },
            "ProcessingOutputConfig": {
                "Outputs": [
                    {
                        "OutputName": name,
                        "S3Output": {
                            "S3Uri": _join(artifact_root, execution, name),
                            "LocalPath": f"/opt/ml/processing/output/{name}",
                            "S3UploadMode": "EndOfJob",
                        },
                    }
                    for name in ("lineage", "evaluation", "model")
                ],
            },
            "StoppingCondition": {"MaxRuntimeInSeconds": 1200},
        },
        "PropertyFiles": [
            {"PropertyFileName": "Evaluation", "OutputName": "evaluation", "FilePath": "evaluation.json"},
        ],
    }
    gate_result = {
        "Std:JsonGet": {
            "PropertyFile": {"Get": "Steps.TrainEvaluate.PropertyFiles.Evaluation"},
            "Path": "passed",
        },
    }
    register = {
        "Name": "RegisterPassingModel",
        "Type": "RegisterModel",
        "Arguments": {
            "ModelPackageGroupName": {"Get": "Parameters.ModelGroup"},
            "ModelApprovalStatus": "PendingManualApproval",
            "InferenceSpecification": {
                "Containers": [{
                    "Image": image_uri,
                    "ModelDataUrl": _join(artifact_root, execution, "model", "model.tar.gz"),
                }],
                "SupportedContentTypes": ["application/json"],
                "SupportedResponseMIMETypes": ["application/json"],
                "SupportedRealtimeInferenceInstanceTypes": ["ml.m5.large"],
            },
            "ModelMetrics": {
                "ModelQuality": {
                    "Statistics": {
                        "ContentType": "application/json",
                        "S3Uri": _join(artifact_root, execution, "evaluation", "evaluation.json"),
                    },
                },
            },
        },
    }
    publish = {
        "Name": "QualityGate",
        "Type": "Condition",
        "DependsOn": ["TrainEvaluate"],
        "Arguments": {
            "Conditions": [{"Type": "Equals", "LeftValue": gate_result, "RightValue": True}],
            "IfSteps": [register],
            "ElseSteps": [],
        },
    }
    validate_release = {
        "Name": "StagingValidation",
        "Type": "Condition",
        "DependsOn": ["TrainEvaluate"],
        "Arguments": {
            "Conditions": [
                {"Type": "Equals", "LeftValue": gate_result, "RightValue": False},
                {"Type": "Equals", "LeftValue": {"Get": "Parameters.RequireApproval"}, "RightValue": True},
            ],
            "IfSteps": [{
                "Name": "FailRejectedStagingCandidate",
                "Type": "Fail",
                "Arguments": {"ErrorMessage": "The staging candidate failed its offline MAE gate."},
            }],
            "ElseSteps": [],
        },
    }
    return {"Version": "2020-12-01", "Parameters": parameters, "Steps": [processing, publish, validate_release]}


def _join(*parts: str | dict) -> dict:
    return {"Std:Join": {"On": "/", "Values": list(parts)}}
