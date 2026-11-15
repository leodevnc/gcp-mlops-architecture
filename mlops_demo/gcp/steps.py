from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from google.api_core.exceptions import Conflict, NotFound
from google.cloud import aiplatform, storage

from ..schemas import Evaluation
from ..workflow import evaluate, load_approved_model, prepare, train
from .gcs import download, publish


def prepare_dataset(source_uri: str, prepared_uri: str, seed: int) -> dict[str, str | int]:
    client = storage.Client()
    with TemporaryDirectory() as directory:
        source = Path(directory) / "source.csv"
        prepared = Path(directory) / "prepared.json"
        generation = download(client, source_uri, source, pin_generation=True)
        data = prepare(source, prepared, seed=seed)
        publish(client, prepared, prepared_uri)
    return {
        "source_uri": source_uri,
        "source_generation": str(generation),
        "source_sha256": data.source_sha256,
        "split_seed": data.split_seed,
        "training_rows": len(data.train),
        "holdout_rows": len(data.holdout),
    }


def train_candidate(prepared_uri: str, candidate_uri: str) -> None:
    client = storage.Client()
    with TemporaryDirectory() as directory:
        prepared = Path(directory) / "prepared.json"
        candidate = Path(directory) / "model.json"
        download(client, prepared_uri, prepared)
        train(prepared, candidate)
        publish(client, candidate, candidate_uri)


def evaluate_candidate(
    prepared_uri: str, candidate_uri: str, report_uri: str, max_mae: float
) -> Evaluation:
    client = storage.Client()
    with TemporaryDirectory() as directory:
        prepared = Path(directory) / "prepared.json"
        candidate = Path(directory) / "model.json"
        report = Path(directory) / "evaluation.json"
        download(client, prepared_uri, prepared)
        download(client, candidate_uri, candidate)
        assessment = evaluate(prepared, candidate, report, max_mae=max_mae)
        publish(client, report, report_uri)
    return assessment


def register_approved_model(
    candidate_uri: str,
    report_uri: str,
    bundle_uri: str,
    serving_image: str,
    *,
    project: str,
    region: str,
    model_id: str,
) -> dict[str, str]:
    client = storage.Client(project=project)
    with TemporaryDirectory() as directory:
        candidate = Path(directory) / "model.json"
        report = Path(directory) / "evaluation.json"
        download(client, candidate_uri, candidate)
        download(client, report_uri, report)
        load_approved_model(candidate, report)
        alias = "bundle-" + sha256(
            b"\0".join((candidate.read_bytes(), report.read_bytes(), serving_image.encode()))
        ).hexdigest()
        publish(client, candidate, f"{bundle_uri.rstrip('/')}/{candidate.name}")
        publish(client, report, f"{bundle_uri.rstrip('/')}/{report.name}")

    model_name = f"projects/{project}/locations/{region}/models/{model_id}"
    context = {"project": project, "location": region}
    for attempt in range(2):
        version = _find_model(f"{model_name}@{alias}", **context)
        if version is not None:
            break

        parent = _find_model(model_name, **context)
        destination = {"model_id": model_id} if parent is None else {"parent_model": model_name}
        try:
            version = aiplatform.Model.upload(
                display_name=model_id,
                artifact_uri=bundle_uri,
                serving_container_image_uri=serving_image,
                serving_container_health_route="/health",
                serving_container_predict_route="/predict",
                serving_container_ports=[8080],
                version_aliases=[alias],
                is_default_version=parent is None,
                **context,
                **destination,
            )
        except Conflict:
            # The registry group or alias may have been created by a concurrent run.
            if attempt:
                raise
        else:
            break

    return {
        "resourceName": version.resource_name,
        "versioned_resource_name": version.versioned_resource_name,
        "version_alias": alias,
    }


def _find_model(name: str, *, project: str, location: str) -> aiplatform.Model | None:
    try:
        return aiplatform.Model(name, project=project, location=location)
    except NotFound:
        return None
