import json
import shutil
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

import boto3

from ..schemas import Evaluation
from ..workflow import evaluate, prepare, train


def split_s3_uri(uri: str) -> tuple[str, str]:
    bucket, separator, key = uri.removeprefix("s3://").partition("/")
    if not uri.startswith("s3://") or not bucket or not separator or not key:
        raise ValueError("Use an s3://bucket/key URI.")
    return bucket, key


def process_s3(source_uri: str, output: Path, *, seed: int, max_mae: float) -> Evaluation:
    bucket, key = split_s3_uri(source_uri)
    response = boto3.client("s3").get_object(Bucket=bucket, Key=key)
    with TemporaryDirectory() as directory:
        source = Path(directory) / "source.csv"
        with response["Body"] as body, source.open("wb") as target:
            shutil.copyfileobj(body, target)
        return build_artifacts(
            source,
            output,
            seed=seed,
            max_mae=max_mae,
            source_metadata={
                "source_uri": source_uri,
                "source_version_id": response.get("VersionId"),
                "source_etag": response.get("ETag"),
            },
        )


def build_artifacts(
    source: Path,
    output: Path,
    *,
    seed: int,
    max_mae: float,
    source_metadata: dict[str, str | None],
) -> Evaluation:
    prepared = output / "lineage" / "prepared.json"
    candidate = output / "lineage" / "model.json"
    report = output / "evaluation" / "evaluation.json"
    bundle = output / "model" / "model.tar.gz"
    data = prepare(source, prepared, seed=seed)
    train(prepared, candidate)
    assessment = evaluate(prepared, candidate, report, max_mae=max_mae)

    lineage = {
        **source_metadata,
        "source_sha256": data.source_sha256,
        "split_seed": seed,
        "train_rows": len(data.train),
        "holdout_rows": len(data.holdout),
    }
    (prepared.parent / "manifest.json").write_text(
        json.dumps(lineage, indent=2) + "\n",
        encoding="utf-8",
    )
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, "w:gz") as archive:
        archive.add(candidate, arcname=candidate.name)
        archive.add(report, arcname=report.name)
    return assessment
