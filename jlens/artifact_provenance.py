"""Verify downloaded lens bytes, not the scientific provenance of their fitting."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


@dataclass(frozen=True)
class HubLensArtifact:
    """Receipt bound to an immutable Hub revision and its published LFS digest."""

    repo_id: str
    filename: str
    requested_revision: str
    resolved_revision: str
    sha256: str
    size: int
    path: str

    def verify(self) -> Path:
        """Recheck the selected file; never search for similarly named local files."""
        path = Path(self.path)
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        if size != self.size or digest.hexdigest() != self.sha256:
            raise ValueError(f"Hub lens size/SHA-256 mismatch: {path}")
        return path

    def to_dict(self) -> dict:
        """JSON-serializable receipt; fit/model compatibility remains unverified."""
        return {
            **asdict(self),
            "verification": "Hub LFS size and SHA-256 at resolved revision",
            "fitting_provenance": "unverified legacy final lens",
        }


def download_lens_artifact(
    repo_id: str, filename: str, revision: str = "main",
) -> HubLensArtifact:
    """Resolve once, download that exact revision, and verify its LFS size/hash.

    A matching Hub cache entry may be reused, but is still fully hashed. Missing
    metadata, download failures, and corrupt cached bytes stop rather than falling
    back to a local basename. Pin ``revision`` to the recorded commit for reruns.
    The Hub is the trust source: this does not attest training data or model weights.
    """
    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    commit = info.sha
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Hub did not provide an immutable repository commit.")
    entry = next((s for s in info.siblings if s.rfilename == filename), None)
    lfs = getattr(entry, "lfs", None)
    sha256 = getattr(lfs, "sha256", None)
    size = getattr(lfs, "size", None)
    if (
        not isinstance(sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", sha256)
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 0
    ):
        raise ValueError("Selected lens lacks published LFS SHA-256/size metadata.")
    path = hf_hub_download(repo_id=repo_id, filename=filename, revision=commit)
    artifact = HubLensArtifact(
        repo_id, filename, revision, commit, sha256, size,
        str(Path(path).absolute()),
    )
    artifact.verify()
    return artifact
