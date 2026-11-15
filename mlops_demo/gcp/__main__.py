import argparse
import json
import math
import re
from pathlib import Path

from google.api_core.exceptions import GoogleAPIError
from google.auth.exceptions import GoogleAuthError
from google.cloud import aiplatform

from .pipeline import compile_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile or run the continuous training pipeline.")
    commands = parser.add_subparsers(dest="command", required=True)
    compile_parser = commands.add_parser("compile", help="Compile the KFP pipeline definition.")
    compile_parser.add_argument("--image-uri", required=True)
    compile_parser.add_argument("--output", type=Path, required=True)

    submit = commands.add_parser("submit", help="Submit a one-off pipeline job.")
    schedule = commands.add_parser("schedule", help="Create a recurring pipeline schedule.")
    for command in (submit, schedule):
        command.add_argument("--template", type=Path, required=True)
        command.add_argument("--project", required=True)
        command.add_argument("--region", required=True)
        command.add_argument("--pipeline-root", required=True)
        command.add_argument("--source-uri", required=True)
        command.add_argument("--service-account", required=True)
        command.add_argument("--model-id", default="delivery-time")
        command.add_argument("--max-mae", type=float, default=5.0)
        command.add_argument("--seed", type=int, default=42)
    submit.add_argument("--wait", action="store_true")
    schedule.add_argument("--name", required=True)
    schedule.add_argument("--cron", required=True)

    args = parser.parse_args()
    try:
        if args.command == "compile":
            compile_pipeline(args.image_uri, args.output)
            print(args.output)
            return

        if not re.fullmatch(r"gs://[^/\s]+/.+", args.source_uri) or args.source_uri.endswith("/"):
            raise ValueError("The source must be a gs:// URI pointing to a CSV object.")
        if not re.fullmatch(r"gs://[^/\s]+(?:/.*)?", args.pipeline_root):
            raise ValueError("The pipeline root must be a gs:// bucket or prefix.")
        if not re.fullmatch(r"[a-z_][a-z0-9_-]{0,62}", args.model_id):
            raise ValueError("The model ID must be valid for Vertex AI Model Registry.")
        if not math.isfinite(args.max_mae) or args.max_mae < 0:
            raise ValueError("The MAE limit must be a finite, non-negative number.")

        job = aiplatform.PipelineJob(
            display_name="delivery-time-continuous-training",
            template_path=str(args.template),
            project=args.project,
            location=args.region,
            pipeline_root=args.pipeline_root,
            parameter_values={
                "project": args.project,
                "region": args.region,
                "source_uri": args.source_uri,
                "model_id": args.model_id,
                "max_mae": args.max_mae,
                "seed": args.seed,
            },
            enable_caching=False,
        )
        if args.command == "submit":
            job.submit(service_account=args.service_account)
            print(job.resource_name)
            if args.wait:
                job.wait()
            return

        existing = aiplatform.PipelineJobSchedule.list(
            project=args.project,
            location=args.region,
            filter=f"display_name={json.dumps(args.name)}",
        )
        if existing:
            resources = ", ".join(item.resource_name for item in existing)
            raise ValueError(f"A schedule named {args.name!r} already exists at {resources}.")
        recurring = job.create_schedule(
            display_name=args.name,
            cron=args.cron,
            service_account=args.service_account,
            max_concurrent_run_count=1,
            max_concurrent_active_run_count=1,
            allow_queueing=False,
        )
        print(recurring.resource_name)
    except (ValueError, OSError, RuntimeError, GoogleAPIError, GoogleAuthError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
