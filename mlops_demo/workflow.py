import csv
import io
import math
import random
from hashlib import sha256
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory

from pydantic import BaseModel
from sklearn.linear_model import LinearRegression

from .schemas import DeliveryRecord, Evaluation, LinearModel, PreparedData


def prepare(source: Path, output: Path, *, seed: int) -> PreparedData:
    raw = source.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""), strict=True)
    if sorted(reader.fieldnames or []) != sorted(DeliveryRecord.model_fields):
        raise ValueError("CSV must contain distance_km, package_count, and delivery_minutes.")

    records = []
    for row in reader:
        if None in row or None in row.values():
            raise ValueError(f"CSV line {reader.line_num} has the wrong number of columns.")
        try:
            records.append(
                DeliveryRecord(
                    distance_km=float(row["distance_km"]),
                    package_count=int(row["package_count"]),
                    delivery_minutes=float(row["delivery_minutes"]),
                )
            )
        except ValueError as exc:
            raise ValueError(f"CSV line {reader.line_num}: {exc}") from exc

    if len(records) < 12:
        raise ValueError("At least 12 delivery records are required.")
    random.Random(seed).shuffle(records)
    holdout_size = math.ceil(len(records) / 4)
    prepared = PreparedData(
        source_sha256=sha256(raw).hexdigest(),
        split_seed=seed,
        train=records[holdout_size:],
        holdout=records[:holdout_size],
    )
    _write_artifact(output, prepared)
    return prepared


def train(prepared_path: Path, output: Path) -> LinearModel:
    prepared, prepared_sha256 = _read_artifact(prepared_path, PreparedData)
    features = [record.vector() for record in prepared.train]
    labels = [record.delivery_minutes for record in prepared.train]
    estimator = LinearRegression().fit(features, labels)
    distance_coefficient, package_coefficient = estimator.coef_
    model = LinearModel(
        prepared_sha256=prepared_sha256,
        intercept=estimator.intercept_,
        distance_coefficient=distance_coefficient,
        package_coefficient=package_coefficient,
        baseline_minutes=fmean(labels),
    )
    _write_artifact(output, model)
    return model


def evaluate(
    prepared_path: Path, model_path: Path, output: Path, *, max_mae: float
) -> Evaluation:
    if not math.isfinite(max_mae) or max_mae < 0:
        raise ValueError("The MAE limit must be a finite, non-negative number.")

    prepared, prepared_sha256 = _read_artifact(prepared_path, PreparedData)
    model, model_sha256 = _read_artifact(model_path, LinearModel)
    if model.prepared_sha256 != prepared_sha256:
        raise ValueError("The model was trained with a different prepared dataset.")

    mae = fmean(
        abs(model.predict(record) - record.delivery_minutes) for record in prepared.holdout
    )
    baseline_mae = fmean(
        abs(model.baseline_minutes - record.delivery_minutes) for record in prepared.holdout
    )
    evaluation = Evaluation(
        model_sha256=model_sha256,
        holdout_rows=len(prepared.holdout),
        mae=mae,
        baseline_mae=baseline_mae,
        max_mae=max_mae,
        passed=mae <= max_mae and mae < baseline_mae,
    )
    _write_artifact(output, evaluation)
    return evaluation


def load_approved_model(model_path: Path, report_path: Path) -> LinearModel:
    model, model_sha256 = _read_artifact(model_path, LinearModel)
    report, _ = _read_artifact(report_path, Evaluation)
    if report.model_sha256 != model_sha256:
        raise ValueError("The evaluation report belongs to a different model artifact.")
    if not report.passed:
        raise ValueError("The model failed its offline quality gate.")
    return model


def _read_artifact[T: BaseModel](path: Path, schema: type[T]) -> tuple[T, str]:
    content = path.read_bytes()
    return schema.model_validate_json(content), sha256(content).hexdigest()


def _write_artifact(path: Path, artifact: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        temporary.write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
