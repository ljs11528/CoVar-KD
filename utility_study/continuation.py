"""Audited resource amendments with byte-preserving reuse of completed cases."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
import shutil

from .io import read_json, sha256, source_hashes, write_json


RESOURCE_FIELDS = {"created_at", "source_hashes", "allowed_gpus", "max_gpus",
                   "resource_policy", "continuation"}


def scientific_contract(registration):
    return {key: value for key, value in registration.items() if key not in RESOURCE_FIELDS}


def amend(parent_out, out, gpus, resource_policy):
    parent_out, out = Path(parent_out).resolve(), Path(out).resolve()
    if (out / "registration.json").exists():
        raise FileExistsError("A continuation registration cannot be overwritten")
    parent_path = parent_out / "registration.json"
    parent = read_json(parent_path)
    if "continuation" in parent:
        raise ValueError("Nested continuation requires an explicit lineage audit")
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("Repeated or empty GPU allocation")
    current_sources = source_hashes()
    for name, digest in parent["source_hashes"].items():
        if name not in {"runner.py", "workflow.py"} and current_sources.get(name) != digest:
            raise RuntimeError("Protected numerical or input source changed: " + name)
    out.mkdir(parents=True, exist_ok=True)
    inherited = {}
    # Preserve every available observation, including divergence, without
    # selecting by score or replacing the registration hash in an old record.
    for phase in ("smoke", "one_step", "rollout"):
        for source in sorted((parent_out / phase).rglob("*.json")):
            relative = source.relative_to(parent_out)
            target = out / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            inherited[str(relative)] = sha256(target)
    parent_copy = out / "inheritance" / "parent_registration.json"
    parent_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(parent_path, parent_copy)
    manifest = out / "inheritance" / "records.json"
    write_json(manifest, {"parent_output": str(parent_out), "files": inherited})
    registered = copy.deepcopy(parent)
    registered.update(created_at=datetime.now(timezone.utc).isoformat(), source_hashes=current_sources,
                      allowed_gpus=gpus, max_gpus=len(gpus), resource_policy=resource_policy,
                      continuation={"parent_registration_sha256": sha256(parent_copy),
                                    "parent_registration": str(parent_copy),
                                    "records_manifest": str(manifest),
                                    "records_manifest_sha256": sha256(manifest),
                                    "reason": "User authorized coexistence with GPU0/1 keepalive programs; preserve them"})
    assert scientific_contract(registered) == scientific_contract(parent)
    write_json(out / "registration.json", registered)
    validate_continuation(registered, out)
    return registered


def validate_continuation(registered, out):
    lineage = registered.get("continuation")
    if lineage is None:
        return
    parent = Path(lineage["parent_registration"])
    manifest = Path(lineage["records_manifest"])
    if sha256(parent) != lineage["parent_registration_sha256"]:
        raise RuntimeError("Parent registration changed")
    if sha256(manifest) != lineage["records_manifest_sha256"]:
        raise RuntimeError("Inherited record manifest changed")
    if scientific_contract(read_json(parent)) != scientific_contract(registered):
        raise RuntimeError("Resource amendment changed the scientific contract")
    for relative, digest in read_json(manifest)["files"].items():
        target = (Path(out) / relative).resolve()
        if not target.is_relative_to(Path(out).resolve()) or sha256(target) != digest:
            raise RuntimeError("Inherited record changed: " + relative)


def accepts_record(path, row, registered, registration_hash, out):
    if row.get("registration_sha256") == registration_hash:
        return True
    lineage = registered.get("continuation", {})
    if row.get("registration_sha256") != lineage.get("parent_registration_sha256"):
        return False
    manifest = Path(lineage["records_manifest"])
    if sha256(manifest) != lineage["records_manifest_sha256"]:
        raise RuntimeError("Inherited record manifest changed")
    relative = str(Path(path).relative_to(out))
    return read_json(manifest)["files"].get(relative) == sha256(path)


def completed_job(out, phase, dataset, seed, stage, cohorts, registered, registration_hash):
    base = Path(out) / phase / dataset
    cohort_list = cohorts.split(",")
    summary_path = base / f"summary_{'_'.join(cohort_list)}_seed{seed}_stage{stage}.json"
    if not summary_path.is_file():
        return False
    summary = read_json(summary_path)
    horizon = 3 if phase == "smoke" else 128 if phase == "rollout" else 1
    if (summary.get("status") != "PASS" or summary.get("horizon") != horizon
            or summary.get("branches") != len(cohort_list) * len(registered["modes"]) * len(registered["conditions"])
            or not accepts_record(summary_path, summary, registered, registration_hash, Path(out))):
        raise RuntimeError("Existing case summary failed continuation audit")
    for cohort in cohort_list:
        run = next(r for r in registered["datasets"][dataset]["runs"] if r["condition"] == cohort and r["seed"] == seed)
        checkpoint_hash = registered["datasets"][dataset]["input_sha256"][run["checkpoints"][str(stage)]]
        for mode in registered["modes"]:
            for condition in registered["conditions"]:
                path = base / cohort / str(seed) / str(stage) / mode / (condition + ".json")
                row = read_json(path)
                expected = {"dataset": dataset, "seed": seed, "checkpoint": stage, "cohort": cohort,
                            "mode": mode, "condition": condition, "horizon": horizon,
                            "checkpoint_sha256": checkpoint_hash, "state_restored": True}
                if (any(row.get(k) != v for k, v in expected.items()) or row.get("status") not in ("PASS", "DIVERGED")
                        or not accepts_record(path, row, registered, registration_hash, Path(out))):
                    raise RuntimeError("Existing scientific observation failed continuation audit")
    return True
