"""Attest copied inputs and remap execution paths without rewriting provenance."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import platform
import subprocess

from .io import read_json, sha256, source_hashes, write_json
from .register import canonical_files, validate_checkpoint_metadata


def dataset_paths(old, dataset, root, transfers):
    """Map only known input paths; original checkpoint args stay byte-for-byte logical values."""
    runtime = root / Path(old["runtime"]).name
    data = root / "shared/dataset" / ("VOCAug" if dataset == "voc" else "Cityscapes")
    teacher = root / "shared/data/winycg/cirkd/teachers" / Path(old["teacher"]).name
    mapping = {old["teacher"]: str(teacher)}
    for name in ("train_list", "val_list"):
        mapping[old[name]] = str(runtime / Path(old[name]).relative_to(old["runtime"]))
    for name in old["input_sha256"]:
        path = Path(name)
        if path.is_relative_to(old["runtime"]):
            mapping[name] = str(runtime / path.relative_to(old["runtime"]))
    result = copy.deepcopy(old)
    result.update(runtime=str(runtime), data=str(data), teacher=str(teacher),
                  train_list=mapping[old["train_list"]], val_list=mapping[old["val_list"]])
    for original, run in zip(old["runs"], result["runs"]):
        run["origin_checkpoints"] = original["checkpoints"]
        run["checkpoints"] = {}
        for stage in (20000, 40000, 60000):
            name = original["checkpoints"][str(stage)]
            item = transfers[name]
            if item["sha256"] != old["input_sha256"][name]:
                raise RuntimeError("Transfer manifest differs from the registered checkpoint")
            mapping[name] = str(root / item["relative"])
            run["checkpoints"][str(stage)] = mapping[name]
    for original, item in zip(old["monitor"], result["monitor"]):
        for key in ("img", "label"):
            item[key] = str(data / Path(original[key]).relative_to(old["data"]))
    if set(old["input_sha256"]) - set(mapping):
        raise ValueError("An audited input has no explicit destination")
    result["input_sha256"] = {mapping[p]: digest for p, digest in old["input_sha256"].items()}
    result["input_hash_provenance"] = {mapping[p]: why for p, why in old["input_hash_provenance"].items()}
    result["input_origins"] = {mapping[p]: p for p in old["input_sha256"]}
    return result


def migrate(origin, manifest, root, out, gpus, defer_checkpoints=False):
    import torch
    import numpy
    import cv2
    import PIL

    origin, root, out = Path(origin).resolve(), Path(root).resolve(), Path(out).resolve()
    if (out / "registration.json").exists():
        raise RuntimeError("Registration is immutable; use a new output directory")
    previous = read_json(origin)
    transfers = {item["source"]: item for item in read_json(manifest)["files"]}
    allocation = [int(x) for x in gpus.split(",")]
    if not allocation or len(set(allocation)) != len(allocation) or len(allocation) > 4 or min(allocation) < 0:
        raise ValueError("Invalid GPU allocation")
    record = copy.deepcopy(previous)
    record.update(schema_version=2, created_at=datetime.now(timezone.utc).isoformat(),
                  source_registration=str(origin), source_registration_sha256=sha256(origin),
                  origin_source_hashes=previous["source_hashes"], source_hashes=source_hashes(),
                  max_gpus=len(allocation), allowed_gpus=allocation,
                  hardware_cohort="maker_rtx4090_20261015", full_training_runs_launched=0)
    record["migration"] = {
        "purpose": "Rerun Stage A on Maker; do not pool old and new hardware observations",
        "unchanged": ["checkpoint bytes and original args", "monitor IDs and bytes", "paired update streams",
                      "batch sizes", "optimizer", "FP32/TF32 policy", "primary gate"],
        "resource_policy": "Only explicitly allocated idle GPUs; one independent seed job per card",
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                        "numpy": numpy.__version__, "opencv": cv2.__version__, "pillow": PIL.__version__,
                        "gpu_inventory": subprocess.check_output(
                            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"], text=True)},
    }
    pending, verified = [], []
    for dataset, old in previous["datasets"].items():
        config = dataset_paths(old, dataset, root, transfers)
        checkpoint_names = {p for run in config["runs"] for p in run["checkpoints"].values()}
        for name, expected in config["input_sha256"].items():
            if defer_checkpoints and name in checkpoint_names and not Path(name).exists():
                pending.append({"path": name, "expected_origin_sha256": expected})
                continue
            if sha256(name) != expected:
                raise RuntimeError("Copied input hash mismatch: " + name)
            verified.append(name)
        files = canonical_files(dataset, config)
        if len(files) != config["train_count"] or len({f["name"] for f in files}) != len(files):
            raise RuntimeError("Training split count or uniqueness changed")
        if any(not Path(f[key]).is_file() for f in files for key in ("img", "label")):
            raise RuntimeError("Training split contains a missing cached image or label")
        for item in config["monitor"]:
            source = files[item["index"]]
            if any(source[key] != item[key] for key in ("name", "img", "label")):
                raise RuntimeError("Monitoring indices no longer refer to the same examples")
            if sha256(item["img"]) != item["image_sha256"] or sha256(item["label"]) != item["label_sha256"]:
                raise RuntimeError("Monitoring bytes differ from the original registration")
        for run in config["runs"]:
            for stage in record["checkpoints"]:
                name = run["checkpoints"][str(stage)]
                if defer_checkpoints and not Path(name).exists():
                    continue
                state = torch.load(name, map_location="cpu", weights_only=False)
                validate_checkpoint_metadata(state, run, stage)
                del state
        # The top-level helper is imported by the frozen original Trainer.
        config["input_sha256"][str(root / "PCOS.py")] = sha256(root / "PCOS.py")
        record["datasets"][dataset] = config
        print("MIGRATION_DATASET_PASS", dataset, len(files), len(config["monitor"]), flush=True)
    record["migration"]["availability_at_registration"] = {
        "verified": verified, "pending_checkpoint_copies": pending,
        "contract": "All expected hashes fixed from the origin registration; each copy and checkpoint metadata verified before use."
    }
    write_json(out / "registration.json", record)
    print("MIGRATION_REGISTRATION_PASS", sha256(out / "registration.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin-registration", required=True)
    parser.add_argument("--transfer-manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--defer-checkpoints", action="store_true",
                        help="Register fixed origin hashes while copies arrive; runner verifies each state before use")
    args = parser.parse_args()
    migrate(args.origin_registration, args.transfer_manifest, args.root, args.out, args.gpus, args.defer_checkpoints)
