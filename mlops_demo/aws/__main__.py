import argparse
import json
import sys
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from .pipeline import build_pipeline_definition
from .processing import split_s3_uri
from .release import deploy_and_validate, schedule_validated_version


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile or release the SageMaker training pipeline.")
    commands = parser.add_subparsers(dest="command", required=True)
    compile_parser = commands.add_parser("compile")
    release_parser = commands.add_parser("release")
    for command in (compile_parser, release_parser):
        command.add_argument("--image-uri", required=True)
        command.add_argument("--processing-role", required=True)
        command.add_argument("--source-uri", required=True)
        command.add_argument("--artifact-root", required=True)
        command.add_argument("--model-group", required=True)
        command.add_argument("--instance-type", default="ml.m5.large")
    compile_parser.add_argument("--output", required=True, type=Path)
    release_parser.add_argument("--pipeline-name", required=True)
    release_parser.add_argument("--pipeline-role", required=True)
    release_parser.add_argument("--staging-source", required=True)
    release_parser.add_argument("--staging-group", required=True)
    release_parser.add_argument("--manifest-uri", required=True)
    release_parser.add_argument("--revision", required=True)
    release_parser.add_argument("--schedule-name")
    release_parser.add_argument("--schedule-expression", default="")
    release_parser.add_argument("--scheduler-role")
    release_parser.add_argument("--schedule-timezone", default="Asia/Seoul")

    args = parser.parse_args()
    try:
        definition = build_pipeline_definition(
            image_uri=args.image_uri,
            processing_role=args.processing_role,
            source_uri=args.source_uri,
            artifact_root=args.artifact_root,
            model_group=args.model_group,
            instance_type=args.instance_type,
        )
        if args.command == "compile":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(definition, indent=2) + "\n", encoding="utf-8")
            print(args.output)
            return 0

        if args.schedule_expression and not (args.schedule_name and args.scheduler_role):
            raise ValueError("A schedule expression requires a schedule name and a scheduler role.")
        split_s3_uri(args.staging_source)
        manifest_bucket, manifest_key = split_s3_uri(args.manifest_uri)
        session = boto3.Session()
        release = deploy_and_validate(
            session.client("sagemaker"),
            pipeline_name=args.pipeline_name,
            pipeline_role=args.pipeline_role,
            definition=definition,
            staging_source=args.staging_source,
            staging_group=args.staging_group,
        )
        manifest = {**release, "revision": args.revision, "image_uri": args.image_uri}
        payload = json.dumps(manifest, indent=2) + "\n"
        session.client("s3").put_object(
            Bucket=manifest_bucket, Key=manifest_key, Body=payload.encode(), ContentType="application/json"
        )
        if args.schedule_expression:
            schedule_validated_version(
                session.client("scheduler"),
                name=args.schedule_name,
                expression=args.schedule_expression,
                timezone=args.schedule_timezone,
                scheduler_role=args.scheduler_role,
                release=release,
            )
        print(payload, end="")
        return 0
    except (BotoCoreError, ClientError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
