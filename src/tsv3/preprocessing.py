"""고정된 development split의 공유 전처리. CPU·case 단위 압축 cache와 재개를 지원한다."""

import argparse
import fcntl
import importlib.metadata
import json
import os
import shutil
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from .audit import ct_content_hash, geometry_delta_voxels
from .data import ROOT, DEFAULT_RAW, DEFAULT_INVENTORY, load_inputs, read_csv, sha256, write_json, git_provenance
from .review import load_review_ledger, panel_labels
from .splits import make_splits, study_eligibility

DEVELOPMENT_ROLES = ("M0_train", "M1_finetune", "M2_finetune", "adult_validation", "pediatric_validation")


def preprocess_case(raw, case_id, labels, target_spacing=None, *, expected_ct_hash=None, expected_organs=None):
    start = time.perf_counter()
    source = nib.load(raw / case_id / "ct.nii.gz")
    if expected_ct_hash is not None:
        source_values = np.asarray(source.dataobj, dtype=np.float32)
        if ct_content_hash(source_values, source.affine) != expected_ct_hash:
            raise ValueError("CT content differs from frozen QC: " + case_id)
        source = nib.Nifti1Image(source_values, source.affine, source.header)
    image = nib.as_closest_canonical(source)
    values = np.asarray(image.dataobj, dtype=np.float32)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("Invalid CT values/shape")
    segmentation = np.zeros(values.shape, dtype=np.uint8)
    overlap = np.zeros(values.shape, dtype=bool)
    native_counts = []
    spacing = np.sqrt((image.affine[:3, :3] ** 2).sum(axis=0))
    ignore = len(labels) + 1
    for label_id, label in enumerate(labels, 1):
        mask_image = nib.as_closest_canonical(nib.load(raw / case_id / "segmentations" / (label + ".nii.gz")))
        mask_values = np.asanyarray(mask_image.dataobj)
        if not np.isin(mask_values, [0, 1]).all():
            raise ValueError("Nonbinary mask: " + label)
        mask = mask_values.astype(bool)
        if mask.shape != values.shape:
            raise ValueError("QC/canonical geometry mismatch")
        if geometry_delta_voxels(mask_image.affine, image.affine, values.shape, spacing) > 0.1:
            raise ValueError("CT-mask affine mismatch: " + label)
        if mask_image.header.get_xyzt_units()[0] != image.header.get_xyzt_units()[0]:
            raise ValueError("CT-mask units mismatch: " + label)
        native_counts.append(int(np.count_nonzero(mask)))
        if expected_organs is not None:
            expected = expected_organs[label]
            border = any(np.any(face) for face in (mask[0], mask[-1], mask[:, 0], mask[:, -1], mask[:, :, 0], mask[:, :, -1]))
            if native_counts[-1] != int(expected["nonzero_voxels"]) or int(border) != int(expected["touches_border"]):
                raise ValueError("Mask counts/border changed since QC: " + label)
        overlap |= mask & (segmentation != 0)
        segmentation[mask] = label_id
    segmentation[overlap] = ignore
    overlap_count = int(overlap.sum())
    del overlap, mask, mask_values, mask_image
    spacing = np.sqrt((image.affine[:3, :3] ** 2).sum(axis=0))
    new_shape = np.array(values.shape) if target_spacing is None else np.maximum(1, np.round(np.asarray(values.shape) * spacing / target_spacing)).astype(int)
    sample = values[(segmentation > 0) & (segmentation != ignore)]
    rng = np.random.default_rng(0)
    if sample.size > 100_000:
        sample = sample[rng.choice(sample.size, 100_000, replace=False)]
    sample = sample.copy()
    if np.array_equal(new_shape, values.shape):
        data, seg = values[None], segmentation[None]
    else:
        from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape
        data = resample_data_or_seg_to_shape(values[None], new_shape, spacing, target_spacing,
                                             is_seg=False, order=1, force_separate_z=False).astype(np.float32)
        seg = resample_data_or_seg_to_shape(segmentation[None], new_shape, spacing, target_spacing,
                                            is_seg=True, order=0, force_separate_z=False).astype(np.uint8)
    target_counts = np.bincount(seg.ravel(), minlength=ignore+1)
    properties = {"case_id": case_id, "original_shape": list(values.shape),
                  "preprocessed_shape": new_shape.tolist(), "original_spacing": spacing.tolist(),
                  "spatial_unit": image.header.get_xyzt_units()[0],
                  "source_shape": list(source.shape), "source_affine": source.affine.tolist(),
                  "canonical_affine": image.affine.tolist(),
                  "overlap_voxels_ignored": overlap_count,
                  "overlap_fraction": overlap_count / segmentation.size,
                  "native_nonempty_labels": sum(n > 0 for n in native_counts),
                  "training_target_nonempty_labels": int(np.count_nonzero(target_counts[1:ignore])),
                  "labels_lost_after_overlap_handling_and_resampling": [
                      label for i, (label, count) in enumerate(zip(labels, native_counts), 1)
                      if count > 0 and target_counts[i] == 0],
                  "preprocessing_seconds": time.perf_counter() - start}
    return data, seg, sample, properties, image


def fit_normalization(samples, fit_cases):
    values = np.concatenate(samples)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Invalid normalization samples")
    result = {"fit_cases": sorted(fit_cases), "lower": float(np.percentile(values, 0.5)),
              "upper": float(np.percentile(values, 99.5)), "mean": float(values.mean()), "std": float(values.std())}
    if result["std"] <= 0 or result["lower"] >= result["upper"]:
        raise ValueError("Degenerate adult-only normalization")
    return result


def normalize_inplace(data, normalization):
    np.clip(data, normalization["lower"], normalization["upper"], out=data)
    data -= normalization["mean"]
    data /= normalization["std"]


def foreground_locations(seg, label_count, seed, limit):
    """매 update의 전체 mask 탐색을 피하기 위한 장기별 좌표 표본."""
    rng = np.random.default_rng(seed)
    result = []
    for label_id in range(1, label_count + 1):
        flat = np.flatnonzero(seg[0] == label_id)
        if not flat.size:
            raise ValueError("Positive organ disappeared from training target")
        if len(flat) > limit:
            flat = flat[rng.choice(len(flat), limit, replace=False)]
        xyz = np.stack(np.unravel_index(flat, seg.shape[1:]), axis=1)
        result.append(np.column_stack((np.full(len(xyz), label_id), xyz)).astype(np.int32))
    return np.concatenate(result)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def check_space(output, host_volume, needed_bytes, reserve_gib):
    roots = [output]
    if host_volume is not None:
        roots.append(host_volume)
    free = min(shutil.disk_usage(p).free for p in roots)
    if free < needed_bytes + reserve_gib * 2**30:
        raise OSError(f"Insufficient free space: {free / 2**30:.2f} GiB; preserve {reserve_gib:g} GiB reserve")
    return free


def load_plan(args):
    study = json.loads(args.study.read_text())
    split_path = ROOT / study["cohort"]["split_manifest"]
    split = json.loads(split_path.read_text())
    if split["status"] != "development_split_frozen_test_organ_eligibility_pending":
        raise ValueError("Expected a frozen development split")
    snapshot = split["study_snapshot"]
    for key in ("cohort", "labels", "qc_protocol"):
        if study[key] != snapshot[key]:
            raise ValueError("Frozen split contract changed: " + key)
    labels = panel_labels(study)
    _, _, _, protocol, provenance = load_inputs(args.raw, DEFAULT_INVENTORY, ROOT / study["qc_protocol"])
    for key, value in provenance.items():
        if split["provenance"][key] != value:
            raise ValueError("Frozen source changed: " + key)
    qc = ROOT / "manifests/qc_v300"
    for path, key in ((qc / "case_qc.csv", "case_qc_sha256"), (qc / "organ_qc.csv", "organ_qc_sha256"),
                      (args.review / "review.csv", "review_csv_sha256"), (args.review / "cases.csv", "review_cases_sha256")):
        if sha256(path) != split["provenance"][key]:
            raise ValueError("Frozen QC/review changed: " + key)
    _, cases, reviews, source_organs = load_review_ledger(study, args.review, qc)
    rows = read_csv(qc / "case_qc.csv")
    cohort = study["cohort"]
    settings = {**protocol, "seed": cohort["split_seed"],
                "finetuning_cases_per_arm": cohort["finetuning_cases_per_arm"],
                "validation_cases_per_cohort": cohort["validation_cases_per_cohort"]}
    screened = study_eligibility(rows, cases, reviews, len(labels))
    if make_splits(screened, settings) != split["partitions"]:
        raise ValueError("Frozen split cannot be reproduced")
    roles = {case: role for role in DEVELOPMENT_ROLES for case in split["partitions"][role]}
    selected_rows = {r["case_id"]: r for r in rows if r["case_id"] in roles}
    execution = study["training"]["fixed_execution_plan"]
    if execution["preprocessing_grid"] != "canonical_RAS_native_grid_no_additional_downsampling":
        raise ValueError("Only the fixed native grid is supported")
    for row in selected_rows.values():
        if row["official_split"] != "train" or not np.allclose(json.loads(row["spacing"]), execution["expected_native_spacing"],
                                                               atol=execution["spacing_absolute_tolerance"], rtol=0):
            raise ValueError("Unexpected development split or native spacing")
    config = study["training"]["preprocessing_cache"]
    if config["format"] != "npz_compressed_float32_CT_uint8_target" or config["normalization_fit"] != "M0_train_only":
        raise ValueError("Unsupported preprocessing cache contract")
    signature = {"schema_version": 1, "split_sha256": sha256(split_path), "labels": study["labels"],
                 "preprocessing": config, "expected_native_spacing": execution["expected_native_spacing"],
                 "spacing_absolute_tolerance": execution["spacing_absolute_tolerance"],
                 "preprocessing_code_sha256": sha256(__file__),
                 "audit_helpers_sha256": sha256(Path(__file__).with_name("audit.py")),
                 "versions": {name: importlib.metadata.version(name) for name in ("numpy", "nibabel")}}
    return {"study": study, "split": split, "signature": signature, "roles": roles,
            "rows": selected_rows, "organs": source_organs, "labels": labels,
            "study_sha256": sha256(args.study), "provenance": provenance}


def source_files(raw, case_id, labels):
    paths = [raw / case_id / "ct.nii.gz", *[raw / case_id / "segmentations" / (label + ".nii.gz") for label in labels]]
    return {str(path.relative_to(raw)): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in paths}


def prepare_one(plan, case_id, raw, output):
    row = plan["rows"][case_id]
    labels = plan["labels"]
    config = plan["signature"]["preprocessing"]
    sources = source_files(raw, case_id, labels)
    data, seg, sample, properties, _ = preprocess_case(
        raw, case_id, labels, expected_ct_hash=row["ct_voxel_sha256"],
        expected_organs={label: plan["organs"][(case_id, label)] for label in labels})
    if not np.allclose(properties["original_spacing"], plan["signature"]["expected_native_spacing"],
                       atol=plan["signature"]["spacing_absolute_tolerance"], rtol=0):
        raise ValueError("Actual native spacing changed")
    if properties["training_target_nonempty_labels"] != len(labels) or properties["labels_lost_after_overlap_handling_and_resampling"]:
        raise ValueError("Label preservation failed: " + case_id)
    locations = foreground_locations(seg, len(labels), config["sampling_seed"], config["locations_per_organ"])
    if plan["roles"][case_id] != "M0_train":
        sample = np.empty(0, dtype=np.float32)
    destination = output / (case_id + ".npz")
    temporary = output / (case_id + ".npz.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, data=data, seg=seg, normalization_sample=sample, foreground_locations=locations)
        # NPZ is lossless for stored float32/uint8 arrays. Reopen the archive to check dtype and dimensions.
        with np.load(temporary, allow_pickle=False) as cached:
            for key, expected in (("data", data), ("seg", seg), ("foreground_locations", locations), ("normalization_sample", sample)):
                restored = cached[key]
                if restored.dtype != expected.dtype or not np.array_equal(restored, expected):
                    raise ValueError("Cache round-trip mismatch: " + key)
                del restored
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"role": plan["roles"][case_id], "file": destination.name, "bytes": destination.stat().st_size,
            "sha256": sha256(destination), "source_files": sources, "properties": properties}


def finish_normalization(plan, output, completed):
    fit_cases = sorted(case for case, role in plan["roles"].items() if role == "M0_train")
    if set(completed) != set(plan["roles"]):
        raise ValueError("Cannot finalize an incomplete development cache")
    samples = []
    for case in fit_cases:
        if completed[case]["role"] != "M0_train":
            raise ValueError("Normalization source is not M0 train")
        with np.load(output / completed[case]["file"], allow_pickle=False) as cached:
            samples.append(cached["normalization_sample"])
    normalization = fit_normalization(samples, fit_cases)
    normalization["method"] = "foreground_max_100000_voxels_per_M0_case_seed_0_percentiles_0.5_99.5_mean_std_before_clip"
    normalization["split_sha256"] = plan["signature"]["split_sha256"]
    atomic_json(output / "normalization.json", normalization)
    return {"file": "normalization.json", "sha256": sha256(output / "normalization.json"), "fit_case_count": len(fit_cases)}


def run_cache(args, plan):
    output = args.output.resolve()
    if output.is_relative_to(args.raw.resolve()) or args.raw.resolve().is_relative_to(output):
        raise ValueError("Cache output must be separate from raw input")
    if output.exists() and not args.resume:
        raise FileExistsError("Cache exists; use --resume or a new output path")
    if args.resume and not (output / "manifest.json").is_file():
        raise FileNotFoundError("No resumable cache manifest")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = output / "manifest.json"
        if args.resume:
            manifest = json.loads(manifest_path.read_text())
            if manifest["signature"] != plan["signature"]:
                raise ValueError("Cache signature changed; use a new version, never mix cached cases")
        else:
            manifest = {"status": "running", "signature": plan["signature"], "completed": {}, "normalization": None,
                        "initial_study_sha256": plan["study_sha256"], "provenance": {**plan["provenance"], **git_provenance()},
                        "expected_cases": len(plan["roles"]), "error": None}
            atomic_json(manifest_path, manifest)
        if not set(manifest["completed"]) <= set(plan["roles"]):
            raise ValueError("Unexpected cases in cache manifest")
        def event(value):
            line = json.dumps(value, ensure_ascii=False)
            print(line, flush=True)
            with (output / "run.log").open("a") as stream:
                stream.write(line + "\n")
        try:
            for case, record in manifest["completed"].items():
                path = output / (case + ".npz")
                if record["file"] != path.name or not path.is_file() or sha256(path) != record["sha256"]:
                    raise ValueError("Completed cache file missing or changed: " + case)
                if source_files(args.raw, case, plan["labels"]) != record["source_files"]:
                    raise ValueError("Raw input changed since caching: " + case)
            pending = sorted(set(plan["roles"]) - set(manifest["completed"]))
            if args.limit:
                pending = pending[:args.limit]
            manifest.update(status="running", error=None)
            atomic_json(manifest_path, manifest)
            for case in pending:
                start = time.monotonic()
                # Conservatively allow one uncompressed case plus persistent free-space reserve.
                needed = int(np.prod(json.loads(plan["rows"][case]["shape"]))) * 5 + 4 * 2**20
                check_space(output, args.host_volume, needed, args.reserve_gib)
                manifest["completed"][case] = prepare_one(plan, case, args.raw, output)
                atomic_json(manifest_path, manifest)
                event({"case_id": case, "completed": len(manifest["completed"]), "expected": len(plan["roles"]),
                       "seconds": round(time.monotonic() - start, 2), "cache_mib": round(manifest["completed"][case]["bytes"] / 2**20, 2)})
            if set(manifest["completed"]) == set(plan["roles"]):
                manifest["normalization"] = finish_normalization(plan, output, manifest["completed"])
                manifest["status"] = "development_cache_complete_not_training_results"
            else:
                manifest["status"] = "partial_limit_not_training_ready"
            manifest["artifact_bytes"] = sum(r["bytes"] for r in manifest["completed"].values())
            atomic_json(manifest_path, manifest)
            event({"status": manifest["status"], "completed": len(manifest["completed"]), "manifest": str(manifest_path)})
        except BaseException as exc:
            manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=type(exc).__name__ + ": " + str(exc))
            atomic_json(manifest_path, manifest)
            event({"status": manifest["status"], "error": manifest["error"]})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "configs/study.json")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--review", type=Path, default=ROOT / "manifests/review_v300")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Smoke test: number of additional cases; 0 means all remaining")
    parser.add_argument("--host-volume", type=Path, help="Also guard free space on the Windows volume hosting the WSL disk")
    parser.add_argument("--reserve-gib", type=float, default=10)
    args = parser.parse_args()
    if args.limit < 0 or not np.isfinite(args.reserve_gib) or args.reserve_gib < 1:
        parser.error("limit must be >=0 and reserve-gib >=1")
    if args.host_volume is not None and not args.host_volume.is_dir():
        parser.error("host-volume must exist")
    plan = load_plan(args)
    if args.output is None:
        args.output = ROOT / "data/preprocessed" / plan["study"]["study_version"]
    if args.dry_run:
        nbytes = sum(int(np.prod(json.loads(r["shape"]))) * 5 for r in plan["rows"].values())
        print(json.dumps({"status": "plan_only_no_cache_written", "cases": len(plan["roles"]),
                          "normalization_cases": len(plan["split"]["partitions"]["M0_train"]),
                          "uncompressed_arrays_gib": nbytes / 2**30, "compressed_size": "unknown_until_measured",
                          "test_cases_included": 0, "output": str(args.output), "host_volume": str(args.host_volume)}, indent=2))
        return
    run_cache(args, plan)


if __name__ == "__main__":
    main()
