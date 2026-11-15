import re
from hashlib import sha256
from pathlib import Path

from kfp.registry import RegistryClient

from .pipeline import compile_pipeline

KFP_REPOSITORY = r"https://[a-z0-9-]+-kfp\.pkg\.dev/[a-z0-9-]+/[a-z0-9_-]+"
TEMPLATE_VERSION = r"sha256:[a-f0-9]{64}"


def publish_pipeline(image_uri: str, repository: str, tag: str, output: Path) -> str:
    repository = repository.rstrip("/")
    if not re.fullmatch(KFP_REPOSITORY, repository):
        raise ValueError("Use the HTTPS URL of a regional Artifact Registry KFP repository.")
    compile_pipeline(image_uri, output)
    expected_version = f"sha256:{sha256(output.read_bytes()).hexdigest()}"
    package, version = RegistryClient(host=repository).upload_pipeline(
        file_name=str(output),
        tags=[tag],
    )
    if version != expected_version or not re.fullmatch(r"[a-z0-9-]+", package):
        raise ValueError("Artifact Registry returned an unexpected pipeline package or version.")
    return f"{repository}/{package}/{version}"


def validate_template(path: str) -> str:
    if "://" in path and not re.fullmatch(f"{KFP_REPOSITORY}/[a-z0-9-]+/{TEMPLATE_VERSION}", path):
        raise ValueError("Use a local template path or a digest-pinned Artifact Registry KFP URL.")
    return path
