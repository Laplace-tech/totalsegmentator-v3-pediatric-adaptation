"""기존 inventory로 NIfTI를 검사한다. SQLite는 중단 후 재개를 위한 원본 결과다."""

import argparse
import collections
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import itertools
import json
import sqlite3
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from .data import (DEFAULT_RAW, DEFAULT_INVENTORY, DEFAULT_PROTOCOL, ROOT,
                   classify_metadata, git_provenance, load_inputs, sha256, write_csv, write_json)


def geometry(image):
    shape = tuple(int(x) for x in image.shape)
    affine = image.affine
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError("not_nonempty_3d")
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
        raise ValueError("invalid_affine")
    spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("invalid_spacing")
    return shape, spacing


def ct_content_hash(values, affine):
    """감사와 후속 전처리가 동일한 CT content identity를 사용한다."""
    h = hashlib.sha256()
    h.update(np.asarray(values.shape, dtype="<i8").tobytes())
    h.update(np.asarray(np.round(affine, 6), dtype="<f8").tobytes())
    normalized = np.asarray(values, dtype="<f4", order="F")
    h.update(memoryview(normalized.ravel(order="F")).cast("B"))
    return h.hexdigest()


def geometry_delta_voxels(mask_affine, ct_affine, shape, spacing):
    corners = np.array([list(p) + [1] for p in itertools.product(*[(0, n-1) for n in shape])]).T
    return float(np.linalg.norm(((mask_affine - ct_affine) @ corners)[:3], axis=0).max() / np.min(spacing))


def audit_case(task):
    raw, case_id, labels, max_displacement_voxels = task
    start = time.monotonic()
    case = {"case_id": case_id, "technical_errors": [], "flags": [], "ct_voxel_sha256": None}
    organs = []
    path = Path(raw) / case_id
    try:
        ct = nib.load(path / "ct.nii.gz")
        shape, spacing = geometry(ct)
        units = ct.header.get_xyzt_units()[0]
        unit_scale = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}.get(units)
        case.update(shape=list(shape), spacing=spacing.tolist(), spatial_unit=units,
                    ct_compressed_bytes=(path / "ct.nii.gz").stat().st_size)
        if unit_scale is None:
            case["flags"].append("spatial_unit_unknown")
        values = np.asanyarray(ct.dataobj)
        if not np.isfinite(values).all():
            raise ValueError("ct_nonfinite_values")
        case.update(ct_min=float(values.min()), ct_max=float(values.max()))
        if case["ct_min"] == case["ct_max"]:
            raise ValueError("constant_ct")
        case["ct_voxel_sha256"] = ct_content_hash(values, ct.affine)
        del values
        voxel_volume = abs(float(np.linalg.det(ct.affine[:3, :3])))
    except Exception as exc:
        case["technical_errors"].append("ct:" + type(exc).__name__ + ":" + str(exc))
        case["seconds"] = time.monotonic() - start
        return case, organs

    for label in labels:
        row = {"case_id": case_id, "label": label, "error": "", "nonzero_voxels": None,
               "volume_mm3": None, "volume_native_units3": None, "geometry_delta_voxels": None,
               "empty": None, "touches_border": None,
               "annotation_review": "pending"}
        try:
            mask = nib.load(path / "segmentations" / (label + ".nii.gz"))
            mask_shape, _ = geometry(mask)
            if mask_shape != shape:
                raise ValueError("ct_mask_geometry_mismatch")
            displacement = geometry_delta_voxels(mask.affine, ct.affine, shape, spacing)
            row["geometry_delta_voxels"] = displacement
            if displacement > max_displacement_voxels:
                raise ValueError("ct_mask_geometry_mismatch")
            if mask.header.get_xyzt_units()[0] != units:
                raise ValueError("ct_mask_spatial_unit_mismatch")
            arr = np.asanyarray(mask.dataobj)
            if arr.dtype.kind == "f" and not np.isfinite(arr).all():
                raise ValueError("mask_nonfinite_values")
            if np.min(arr) < 0 or np.max(arr) > 1 or (arr.dtype.kind == "f" and not np.equal(arr, np.floor(arr)).all()):
                raise ValueError("mask_not_binary")
            count = int(np.count_nonzero(arr))
            border = bool(any(np.any(face) for face in (
                arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :],
                arr[:, :, 0], arr[:, :, -1])))
            row.update(nonzero_voxels=count, empty=int(count == 0), touches_border=int(border),
                       volume_native_units3=count * voxel_volume,
                       volume_mm3=count * voxel_volume * unit_scale ** 3 if unit_scale is not None else None)
            del arr
        except Exception as exc:
            row["error"] = type(exc).__name__ + ":" + str(exc)
            case["technical_errors"].append("mask:" + label + ":" + row["error"])
        organs.append(row)
    case.update(seconds=time.monotonic() - start,
                masks_checked=len(organs), empty_masks=sum(x["empty"] == 1 for x in organs),
                border_masks=sum(x["touches_border"] == 1 for x in organs))
    return case, organs


def export_results(connection, output, metadata, settings, expected, signature, elapsed):
    cases = [json.loads(row[0]) for row in connection.execute("SELECT payload FROM cases ORDER BY case_id")]
    groups = collections.defaultdict(list)
    for case in cases:
        if case["ct_voxel_sha256"]:
            groups[case["ct_voxel_sha256"]].append(case["case_id"])
    duplicates = [ids for ids in groups.values() if len(ids) > 1]
    duplicate_exclusions = set()
    for group in duplicates:
        # Test는 보존한다. 동일 영상의 train 사본 또는 train-only 중복 그룹은 보류한다.
        duplicate_exclusions.update(x for x in group if metadata[x]["split"] == "train")
    exported = []
    for case in cases:
        case_id = case["case_id"]
        meta = metadata[case_id]
        cohort, reasons = classify_metadata(meta, settings)
        if meta["split"] not in {"train", "test"}:
            reasons.append("unknown_official_split")
        if meta["split"] == "train" and case_id in settings["manual_training_exclusions"]:
            reasons.append(settings["manual_training_exclusions"][case_id])
        if case_id in duplicate_exclusions:
            reasons.append("duplicate_ct_train_quarantine")
        if case["technical_errors"]:
            reasons.append("technical_qc_failure")
        exported.append({
            "case_id": case_id, "official_split": meta["split"], "cohort": cohort,
            "age": meta["age"], "automated_eligible": int(not reasons),
            "exclusion_reasons": ";".join(reasons),
            "technical_errors": json.dumps(case["technical_errors"], ensure_ascii=False),
            "flags": ";".join(case["flags"]), "annotation_review": "pending",
            "patient_group_status": "unavailable", "ct_voxel_sha256": case["ct_voxel_sha256"] or "",
            "shape": json.dumps(case.get("shape")), "spacing": json.dumps(case.get("spacing")),
            "spatial_unit": case.get("spatial_unit", ""), "masks_checked": case.get("masks_checked", 0),
            "empty_masks": case.get("empty_masks", 0), "border_masks": case.get("border_masks", 0),
            "seconds": round(case["seconds"], 3)})
    if exported:
        write_csv(output / "case_qc.csv", exported, list(exported[0]))
    organ_fields = ["case_id", "label", "error", "nonzero_voxels", "volume_mm3", "volume_native_units3", "geometry_delta_voxels", "empty", "touches_border", "annotation_review"]
    write_csv(output / "organ_qc.csv", (json.loads(r[0]) for r in connection.execute(
        "SELECT payload FROM organs ORDER BY case_id,label")), organ_fields)
    counts = collections.Counter((r["official_split"], r["cohort"], r["automated_eligible"]) for r in exported)
    summary = {"status": "automated_complete_review_pending" if len(cases) == expected else "partial",
               "expected_cases": expected, "completed_cases": len(cases),
               "organs_checked": connection.execute("SELECT count(*) FROM organs").fetchone()[0],
               "technical_failure_cases": sum(bool(c["technical_errors"]) for c in cases),
               "duplicate_ct_groups": duplicates,
               "cohort_counts": [{"official_split": k[0], "cohort": k[1], "automated_eligible": k[2], "n": v}
                                 for k, v in sorted(counts.items())],
               "provenance": signature, "elapsed_this_invocation_seconds": elapsed,
               "limits": ["Clinical annotation correctness has not been reviewed.",
                          "No patient identifiers: repeated acquisitions from one patient cannot be ruled out.",
                          "Exact voxel hashes do not detect resampled or near-duplicate CTs.",
                          "Empty and boundary masks are flags, not automatic annotation failures.",
                          "No cohort claim about the officially added 291 pediatric cases."]}
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=ROOT / "manifests/qc_v300")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="Pilot only; incomplete audit cannot be used to freeze splits")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 4:
        parser.error("Use 1-4 workers to bound memory")
    ids, metadata, labels, settings, signature = load_inputs(args.raw, args.inventory, args.protocol)
    signature.update(audit_code_sha256=sha256(__file__), shared_code_sha256=sha256(Path(__file__).with_name("data.py")),
                     nibabel_version=nib.__version__, numpy_version=np.__version__, **git_provenance())
    if args.output.exists() and not args.resume:
        raise FileExistsError("Output exists; use --resume with the same inputs or a new output")
    args.output.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(args.output / "qc.sqlite")
    connection.execute("CREATE TABLE IF NOT EXISTS settings (signature TEXT NOT NULL)")
    connection.execute("CREATE TABLE IF NOT EXISTS cases (case_id TEXT PRIMARY KEY,payload TEXT NOT NULL)")
    connection.execute("CREATE TABLE IF NOT EXISTS organs (case_id TEXT,label TEXT,payload TEXT NOT NULL,PRIMARY KEY(case_id,label))")
    previous = connection.execute("SELECT signature FROM settings").fetchone()
    identity = {k: v for k, v in signature.items() if k not in {"git_dirty", "git_commit"}}
    if previous and json.loads(previous[0]) != identity:
        raise ValueError("Resume provenance differs; choose a new output")
    if not previous:
        connection.execute("INSERT INTO settings VALUES (?)", (json.dumps(identity),))
        connection.commit()
    done = {r[0] for r in connection.execute("SELECT case_id FROM cases")}
    pending = [x for x in ids if x not in done]
    if args.limit:
        pending = pending[:args.limit]
    start = time.monotonic()
    queue = iter(pending)
    completed = len(done)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = set()
        def submit_next():
            item = next(queue, None)
            if item is not None:
                futures.add(pool.submit(audit_case, (str(args.raw), item, labels, settings["max_geometry_displacement_voxels"])))
        for _ in range(args.workers):
            submit_next()
        while futures:
            finished, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in finished:
                futures.remove(future)
                case, organs = future.result()
                with connection:
                    connection.execute("INSERT INTO cases VALUES (?,?)", (case["case_id"], json.dumps(case)))
                    connection.executemany("INSERT INTO organs VALUES (?,?,?)", (
                        (r["case_id"], r["label"], json.dumps(r)) for r in organs))
                completed += 1
                print(json.dumps({"completed": completed, "total": len(ids), "seconds": round(time.monotonic()-start, 1),
                                  "last_case_seconds": round(case["seconds"], 1), "last_case_errors": len(case["technical_errors"])}), flush=True)
                submit_next()
    summary = export_results(connection, args.output, metadata, settings, len(ids), signature, time.monotonic()-start)
    connection.close()
    print(json.dumps({k: v for k, v in summary.items() if k not in {"provenance", "limits"}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
