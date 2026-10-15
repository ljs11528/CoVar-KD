"""Register checkpoints, training-only monitoring images and paired update streams."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import random

from .io import read_json, sha256, source_hashes, write_json

CITY_IDS = (7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33)
COHORTS = ("ce", "shared_c1", "shared_c4")
MODES = ("legacy_eval", "matched_eval", "matched_train")


def validate_checkpoint_metadata(state, run, stage):
    if state.get("iteration") != stage or "student" not in state or "optimizer" not in state:
        raise ValueError("Incomplete checkpoint or incorrect saved iteration")
    for key in ("seed", "dataset", "student_model", "student_backbone", "lambda_kd",
                "kd_loss_mode", "batch_size", "max_iterations", "lr", "teacher_pretrained"):
        if state["args"].get(key) != run["args"].get(key):
            raise ValueError(f"Checkpoint configuration differs from its audited run: {key}")
    if list(state["args"]["crop_size"]) != list(run["args"]["crop_size"]):
        raise ValueError("Checkpoint crop differs from its audited run")


def monitoring_indices(presence, classes, size=256, per_class=8, seed=20261015):
    rng = random.Random(seed)
    selected = set()
    for k in range(classes):
        candidates = [i for i, row in enumerate(presence) if k in row]
        if len(candidates) < per_class:
            raise ValueError(f"Class {k} has fewer than {per_class} train images")
        rng.shuffle(candidates)
        selected.update(candidates[:per_class])
    if len(selected) > size or len(presence) < size:
        raise ValueError("Monitoring pool cannot satisfy the class-coverage contract")
    remainder = [i for i in range(len(presence)) if i not in selected]
    rng.shuffle(remainder)
    selected.update(remainder[:size - len(selected)])
    return sorted(selected)


def canonical_files(dataset, config):
    data = Path(config["data"])
    rows = Path(config["train_list"]).read_text().splitlines()
    if dataset == "voc":
        return [{"name": name.strip(), "img": str(data / "JPEGImages" / (name.strip() + ".jpg")),
                 "label": str(data / "SegmentationClassAug" / (name.strip() + ".png"))} for name in rows]
    result = []
    for row in rows:
        image, label = row.split()
        result.append({"name": Path(label).stem, "img": str(data / image), "label": str(data / label)})
    return result


def label_presence(item, dataset):
    import numpy as np
    from PIL import Image
    with Image.open(item["label"]) as image:
        values = set(int(x) for x in np.unique(np.asarray(image)))
    if dataset == "voc":
        if values - set(range(21)) - {255}:
            raise ValueError("Invalid VOC train label")
        return sorted(values - {255})
    return [k for k, raw in enumerate(CITY_IDS) if raw in values]


def main(root, out):
    root, out = Path(root), Path(out)
    if (out / "registration.json").exists():
        raise RuntimeError("Registration already exists; immutable input preserved")
    source = root / "reports/P13_CrossDataset_Posthoc_20261015/registration.json"
    previous = read_json(source)
    public_sources = read_json(Path(__file__).resolve().parent.parent / "reports/cvpr2027/source_manifest.json")["files"]
    record = {
        "schema_version": 1, "status": "REGISTERED", "created_at": datetime.now(timezone.utc).isoformat(),
        "source_registration": str(source), "source_registration_sha256": sha256(source),
        "source_hashes": source_hashes(), "seeds": [1234, 2025, 3407], "checkpoints": [20000, 40000, 60000],
        "cohorts": list(COHORTS), "modes": list(MODES), "horizons": [1, 128],
        "conditions": ["ce", "shared_c1", "shared_c4", "teacher_c1", "teacher_c4", "shared_norm"],
        "primary": "shared_c4 - shared_c1, pixel_ce on train monitoring pool, CE start, matched_train, horizon1",
        "gate": "positive seed-mean primary contrast at >=2/3 stages in each dataset; no selector training otherwise",
        "sampling_seed": 20261015, "monitor_count": 256, "monitor_min_per_class": 8,
        "precision": "FP32; matmul/cuDNN TF32 off; cuDNN benchmark off", "max_gpus": 2,
        "allowed_gpus": [0, 2], "full_training_runs_launched": 0,
        "limits": ["Monitoring images are from train, and excluded only from virtual updates.",
                   "A virtual rollout does not reproduce a full KD training trajectory.",
                   "Dataset, teacher and training protocols differ together.",
                   "Stages, modes and image pools are not independent training-seed repetitions."],
        "datasets": {},
    }
    for dataset, old in previous["datasets"].items():
        files = canonical_files(dataset, old)
        if len({x["name"] for x in files}) != len(files):
            raise RuntimeError("Duplicate training image names")
        with ThreadPoolExecutor(max_workers=8) as pool:
            presence = list(pool.map(lambda item: label_presence(item, dataset), files))
        indices = monitoring_indices(presence, old["classes"])
        monitor = []
        for i in indices:
            item = dict(files[i], index=i, classes=presence[i])
            item.update(image_sha256=sha256(item["img"]), label_sha256=sha256(item["label"]))
            monitor.append(item)
        excluded = set(indices)
        eligible = [i for i in range(len(files)) if i not in excluded]
        schedules = {}
        for seed in record["seeds"]:
            for stage in record["checkpoints"]:
                for mode_index, mode in enumerate(MODES):
                    batch = 2 if mode == "legacy_eval" else old["runs"][0]["args"]["batch_size"]
                    sampling = 20261015 + seed * 10000 + stage + mode_index * 1000000
                    rng = random.Random(sampling)
                    updates = [{"indices": rng.sample(eligible, batch), "augmentation_seed": sampling + step * 131}
                               for step in range(128)]
                    schedules[f"{seed}_{stage}_{mode}"] = updates
        selected = [r for r in old["runs"] if r["condition"] in COHORTS]
        assert len(selected) == 9
        inputs = {old["teacher"], old["train_list"]}
        checkpoint_inputs = {}
        for run in selected:
            inputs.update(run["checkpoints"][str(k)] for k in record["checkpoints"])
            checkpoint_inputs.update({run["checkpoints"][str(k)]: (run, k) for k in record["checkpoints"]})
        input_sha, hash_provenance = {}, {}
        for name in sorted(inputs):
            digest = sha256(name)
            previous_digest = old["input_sha256"].get(name)
            if previous_digest is not None and digest != previous_digest:
                raise RuntimeError(f"Original input changed: {name}")
            if previous_digest is None:
                if name not in checkpoint_inputs or checkpoint_inputs[name][0]["condition"] == "ce":
                    raise RuntimeError("Unexpected input without a previous hash")
                import torch
                run, stage = checkpoint_inputs[name]
                state = torch.load(name, map_location="cpu", weights_only=False)
                validate_checkpoint_metadata(state, run, stage)
                del state
                hash_provenance[name] = "existing_KD_intermediate_first_hashed_in_P14"
            else:
                hash_provenance[name] = "matched_P13_hash"
            input_sha[name] = digest
        runtime = Path(old["runtime"])
        for path in sorted(runtime.rglob("*.py")):
            rel = path.relative_to(runtime)
            if str(rel) == "train_kd_3090.py" or rel.parts[0] in ("dataset", "models", "losses", "utils"):
                digest = sha256(path)
                public_path = "repro/rtx3090/" + runtime.name + "/" + str(rel)
                if public_path not in public_sources or digest != public_sources[public_path]["sha256"]:
                    raise RuntimeError(f"Runtime differs from the GitHub source snapshot: {path}")
                input_sha[str(path)] = digest
        record["datasets"][dataset] = {
            "runtime": old["runtime"], "data": old["data"], "train_list": old["train_list"],
            "val_list": old["val_list"], "classes": old["classes"], "teacher": old["teacher"],
            "runs": selected, "monitor": monitor, "updates": schedules, "input_sha256": input_sha,
            "input_hash_provenance": hash_provenance,
            "train_count": len(files), "matched_batch": old["runs"][0]["args"]["batch_size"],
            "monitor_class_image_counts": [sum(k in presence[i] for i in indices) for k in range(old["classes"])],
        }
        print(f"REGISTERED_DATASET {dataset} train={len(files)} monitor={len(monitor)}", flush=True)
    write_json(out / "registration.json", record)
    print("REGISTRATION_PASS", sha256(out / "registration.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.root, args.out)
