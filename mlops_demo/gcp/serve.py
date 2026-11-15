import os
from pathlib import Path
from tempfile import TemporaryDirectory

import uvicorn
from google.cloud import storage

from ..serving import create_app
from ..workflow import load_approved_model
from .gcs import download


def main() -> None:
    storage_uri = os.environ["AIP_STORAGE_URI"].rstrip("/")
    client = storage.Client()
    with TemporaryDirectory() as directory:
        candidate = Path(directory) / "model.json"
        report = Path(directory) / "evaluation.json"
        download(client, f"{storage_uri}/{candidate.name}", candidate)
        download(client, f"{storage_uri}/{report.name}", report)
        model = load_approved_model(candidate, report)
    uvicorn.run(create_app(model, vertex=True), host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
