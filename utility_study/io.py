"""Small atomic records and content-addressed experiment inputs."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def source_hashes():
    return {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def verify_sources(expected):
    if source_hashes() != expected:
        raise RuntimeError("Registered source files changed")
