import argparse
from pathlib import Path

import uvicorn

from ..serving import create_batch_app
from ..workflow import load_approved_model
from .processing import process_s3


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a SageMaker processing or prediction container.")
    commands = parser.add_subparsers(dest="command", required=True)
    process = commands.add_parser("process", help="Train and evaluate the model from a CSV in S3.")
    process.add_argument("--source-uri", required=True)
    process.add_argument("--seed", type=int, default=42)
    process.add_argument("--max-mae", type=float, default=5.0)
    process.add_argument("--output-dir", type=Path, default=Path("/opt/ml/processing/output"))
    serve = commands.add_parser("serve", help="Serve the approved model in a SageMaker container.")
    serve.add_argument("--model-dir", type=Path, default=Path("/opt/ml/model"))

    args = parser.parse_args()
    if args.command == "process":
        evaluation = process_s3(
            args.source_uri, args.output_dir, seed=args.seed, max_mae=args.max_mae
        )
        print(evaluation.model_dump_json())
        return

    model = load_approved_model(args.model_dir / "model.json", args.model_dir / "evaluation.json")
    app = create_batch_app(model, health_path="/ping", prediction_path="/invocations")
    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
