import argparse
import csv
from pathlib import Path

import uvicorn

from .serving import create_app
from .workflow import evaluate, load_approved_model, prepare, train


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one step of the local MLOps workflow.")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare", help="Validate and split a CSV dataset.")
    prepare_parser.add_argument("source", type=Path)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--seed", type=int, default=42)

    train_parser = commands.add_parser("train", help="Train a candidate from the training split.")
    train_parser.add_argument("prepared", type=Path)
    train_parser.add_argument("--output", type=Path, required=True)

    evaluate_parser = commands.add_parser("evaluate", help="Validate the candidate on the holdout.")
    evaluate_parser.add_argument("prepared", type=Path)
    evaluate_parser.add_argument("model", type=Path)
    evaluate_parser.add_argument("--report", type=Path, required=True)
    evaluate_parser.add_argument("--max-mae", type=float, required=True)

    serve_parser = commands.add_parser("serve", help="Run the approved model on localhost.")
    serve_parser.add_argument("model", type=Path)
    serve_parser.add_argument("--report", type=Path, required=True)
    serve_parser.add_argument("--port", type=int, default=8080)

    args = parser.parse_args()
    try:
        match args.command:
            case "prepare":
                data = prepare(args.source, args.output, seed=args.seed)
                print(f"Prepared {len(data.train)} training and {len(data.holdout)} holdout rows.")
            case "train":
                train(args.prepared, args.output)
                print(f"Wrote candidate artifact to {args.output}.")
            case "evaluate":
                result = evaluate(args.prepared, args.model, args.report, max_mae=args.max_mae)
                decision = "passed" if result.passed else "failed"
                print(
                    f"Quality gate {decision}. "
                    f"Candidate MAE {result.mae:.3f} min, "
                    f"baseline MAE {result.baseline_mae:.3f} min, "
                    f"limit {result.max_mae:.3f} min."
                )
                return 0 if result.passed else 1
            case "serve":
                model = load_approved_model(args.model, args.report)
                uvicorn.run(create_app(model), host="127.0.0.1", port=args.port)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
