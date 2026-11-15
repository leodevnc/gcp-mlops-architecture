from pathlib import Path

from google.api_core.exceptions import PreconditionFailed
from google.cloud import storage


def download(
    client: storage.Client, uri: str, target: Path, *, pin_generation: bool = False
) -> int | None:
    blob = storage.Blob.from_uri(uri, client=client)
    if pin_generation:
        blob.reload()
    blob.download_to_filename(str(target), if_generation_match=blob.generation)
    return blob.generation


def publish(client: storage.Client, source: Path, uri: str) -> None:
    blob = storage.Blob.from_uri(uri, client=client)
    try:
        blob.upload_from_filename(str(source), if_generation_match=0)
    except PreconditionFailed:
        if blob.download_as_bytes() != source.read_bytes():
            raise ValueError(f"Artifact already exists with different contents at {uri}.")
