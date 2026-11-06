"""기존 QC에서 장기 검토 대상을 만들고 development 영상의 검토용 그림을 생성한다."""

import argparse
import collections
import hashlib
import html
import inspect
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from .data import ROOT, DEFAULT_RAW, DEFAULT_INVENTORY, load_inputs, read_csv, sha256, write_csv, write_json


DEFAULT_STUDY = ROOT / "configs/study.json"
DEFAULT_REVIEW = ROOT / "manifests/review_v300"


def panel_labels(study):
    labels = study["labels"]
    if labels.get("background") != 0 or sorted(labels.values()) != list(range(len(labels))):
        raise ValueError("Class IDs must be unique and consecutive, starting at background=0")
    return [name for name, value in sorted(labels.items(), key=lambda item: item[1]) if value]


def review_configuration(study):
    panel_labels(study)
    if study["cohort"]["review_screen"] != "automated_qc_pass_and_at_least_one_panel_mask_nonempty":
        raise ValueError("Unsupported review screen; prepare a new review implementation/version")
    return {"labels": study["labels"], "display": study["review_display"]}


def validate_review_source(study, review):
    review_configuration(study)
    summary = json.loads((review / "summary.json").read_text())
    if summary["labels"] != study["labels"] or summary["provenance"]["cases_csv_sha256"] != sha256(review / "cases.csv"):
        raise ValueError("Review labels/cases changed; prepare a new version")
    return summary


def screen_cases(cases, organs, labels):
    """자동 screen은 수동 승인과 분리한다. 빈 mask는 부재 원인을 확정하지 않는다."""
    by_case = {r["case_id"]: r for r in cases}
    if len(by_case) != len(cases):
        raise ValueError("Repeated case IDs")
    selected = {}
    for row in organs:
        if row["label"] not in labels:
            continue
        key = (row["case_id"], row["label"])
        if key in selected or row["case_id"] not in by_case:
            raise ValueError("Repeated or unknown case/organ in QC")
        selected[key] = row
    output, reviews = [], []
    counts = collections.defaultdict(collections.Counter)
    for case in cases:
        case_id = case["case_id"]
        rows = [selected.get((case_id, label)) for label in labels]
        if any(row is None for row in rows):
            raise ValueError("Incomplete organ QC: " + case_id)
        present = [r for r in rows if r["empty"] == "0" and not r["error"]]
        interior = [r for r in present if r["touches_border"] == "0"]
        eligible = case["automated_eligible"] == "1"
        candidate = eligible and bool(present)
        reason = case["exclusion_reasons"] if not eligible else ("" if present else "no_nonempty_panel_mask")
        output.append({"case_id": case_id, "official_split": case["official_split"], "cohort": case["cohort"],
                       "automated_eligible": int(eligible), "review_candidate": int(candidate),
                       "nonempty_panel_organs": len(present), "nonempty_no_border_organs": len(interior),
                       "screen_reason": reason, "shape": case["shape"]})
        c = counts[(case["official_split"], case["cohort"])]
        c["total"] += 1
        c["automated_eligible"] += eligible
        c["any_panel_nonempty"] += candidate
        c["all_panel_nonempty"] += eligible and len(present) == len(labels)
        c["all_panel_nonempty_no_border"] += eligible and len(interior) == len(labels)
        if candidate and case["official_split"] == "train":
            for row in rows:
                reviews.append({"case_id": case_id, "label": row["label"], "cohort": case["cohort"],
                                "automatic_empty": row["empty"], "automatic_border": row["touches_border"],
                                "coverage": "pending", "annotation": "pending", "reviewer": "", "notes": ""})
    summary = [{"official_split": key[0], "cohort": key[1], **value} for key, value in sorted(counts.items())]
    return output, reviews, summary


def prepare(args):
    if args.output.exists():
        raise FileExistsError("Review manifest exists; keep manual reviews and use a new version if inputs change")
    study = json.loads(args.study.read_text())
    review_configuration(study)
    labels = panel_labels(study)
    protocol_path = ROOT / study["qc_protocol"]
    ids, _, available_labels, _, provenance = load_inputs(args.raw, DEFAULT_INVENTORY, protocol_path)
    if not set(labels) <= set(available_labels):
        raise ValueError("Selected labels not present in original inventory")
    qc_summary = json.loads((args.qc / "summary.json").read_text())
    if qc_summary["completed_cases"] != len(ids) or qc_summary["status"] != "automated_complete_review_pending":
        raise ValueError("Complete QC required")
    for key in ("metadata_sha256", "inventory_sha256", "labels_sha256", "protocol_sha256"):
        if qc_summary["provenance"][key] != provenance[key]:
            raise ValueError("QC source provenance mismatch: " + key)
    cases = read_csv(args.qc / "case_qc.csv")
    if len(cases) != len(ids) or {r["case_id"] for r in cases} != set(ids):
        raise ValueError("QC case inventory mismatch")
    case_rows, review_rows, counts = screen_cases(cases, read_csv(args.qc / "organ_qc.csv"), labels)
    if not review_rows:
        raise ValueError("No development review candidates")
    args.output.mkdir(parents=True)
    write_csv(args.output / "cases.csv", case_rows, list(case_rows[0]))
    write_csv(args.output / "review.csv", review_rows, list(review_rows[0]))
    summary = {"status": "screen_complete_manual_review_pending_not_split", "labels": study["labels"], "counts": counts,
               "development_review_rows": len(review_rows),
               "provenance": {**provenance, "study_sha256": sha256(args.study), "review_code_sha256": sha256(__file__),
                              "case_qc_sha256": sha256(args.qc / "case_qc.csv"), "organ_qc_sha256": sha256(args.qc / "organ_qc.csv"),
                              "cases_csv_sha256": sha256(args.output / "cases.csv")},
               "manual_review_values": {"coverage": ["pending", "full", "partial", "out_of_fov", "true_absence", "uncertain"],
                                        "annotation": ["pending", "acceptable", "reject", "uncertain"]},
               "limitations": ["Nonempty and non-border masks do not prove anatomical completeness or label accuracy.",
                               "No K or split is selected from these counts.", "Test data are counted for feasibility only; no test review images are generated.",
                               "Manual review.csv is an editable ledger; cases.csv and summary.json are generated immutable outputs."]}
    write_json(args.output / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "counts": counts, "review_rows": len(review_rows)}, ensure_ascii=False, indent=2))


def make_montage(raw, case, labels, display):
    from PIL import Image, ImageDraw
    from scipy.ndimage import binary_erosion

    path = raw / case["case_id"]
    original = nib.load(path / "ct.nii.gz")
    if list(original.shape) != json.loads(case["shape"]):
        raise ValueError("CT shape differs from audited shape")
    transform = nib.orientations.ornt_transform(nib.orientations.io_orientation(original.affine),
                                                nib.orientations.axcodes2ornt(("R", "A", "S")))
    image = nib.as_closest_canonical(original)
    values = np.asarray(image.dataobj, dtype=np.float32)
    spacing = np.sqrt((image.affine[:3, :3] ** 2).sum(axis=0))
    width, height = display["tile_size"]
    row_height = height + 45
    canvas = Image.new("RGB", (width * 3, 55 + row_height * len(labels)), "black")
    draw = ImageDraw.Draw(canvas)
    lo, hi = display["intensity_window"]
    draw.text((8, 5), f"{case['case_id']} | development QC | window [{lo}, {hi}] native intensity", fill="white")
    draw.text((8, 23), "RAS array: sagittal / coronal / axial; bbox center slices; full-volume review still required", fill="white")
    for row_index, label in enumerate(labels):
        mask_img = nib.load(path / "segmentations" / (label + ".nii.gz"))
        if mask_img.shape != original.shape:
            raise ValueError("Mask shape differs from CT")
        mask = nib.orientations.apply_orientation(np.asarray(mask_img.dataobj, dtype=bool), transform)
        bounds = []
        for axis in range(3):
            occupied = np.flatnonzero(mask.any(axis=tuple(i for i in range(3) if i != axis)))
            bounds.append((int(occupied[0]), int(occupied[-1])) if occupied.size else None)
        centers = [(bound[0] + bound[1]) // 2 if bound is not None else mask.shape[i] // 2
                   for i, bound in enumerate(bounds)]
        border = any(bound is not None and (bound[0] == 0 or bound[1] == mask.shape[i] - 1) for i, bound in enumerate(bounds))
        y = 55 + row_index * row_height
        draw.text((8, y), f"{label} | {'EMPTY - cause unknown' if bounds[0] is None else 'nonempty'} | boundary={border}", fill="white")
        for axis, index in enumerate(centers):
            ct_slice = np.rot90(np.take(values, index, axis=axis))
            mask_slice = np.rot90(np.take(mask, index, axis=axis))
            gray = (np.clip((ct_slice - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
            rgb = np.repeat(gray[..., None], 3, axis=-1)
            contour = mask_slice & ~binary_erosion(mask_slice)
            rgb[contour] = (255, 70, 40)
            axes = [i for i in range(3) if i != axis]
            physical = np.array([ct_slice.shape[1] * spacing[axes[0]], ct_slice.shape[0] * spacing[axes[1]]])
            factor = min(width / physical[0], height / physical[1])
            size = tuple(np.maximum(1, np.round(physical * factor).astype(int)))
            tile = Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR)
            x = axis * width
            canvas.paste(tile, (x + (width - size[0]) // 2, y + 20 + (height - size[1]) // 2))
            orientation = ("Sag P>A top S", "Cor L>R top S", "Ax L>R top A")[axis]
            draw.text((x + 8, y + 22 + height), f"{orientation} | slice {index}", fill="white")
        del mask, mask_img
    return canvas


def write_index(output, completed):
    body = ["<!doctype html><meta charset='utf-8'><title>Development organ QC</title>",
            "<style>body{background:#222;color:#eee;font:16px sans-serif;max-width:1000px;margin:auto}img{max-width:100%}a{color:#9cf}</style>",
            "<h1>Development 장기 검토</h1><p>자동 승인 결과가 아닙니다. 각 장기의 bbox 중심 3개 단면이며, 전체 촬영 범위와 annotation 판정에는 원본 전체 영상 확인이 필요합니다.</p>",
            "<p>판정은 이 실행에 사용한 review manifest의 review.csv에 기록합니다. test 영상은 포함하지 않습니다.</p>"]
    for case_id in sorted(completed):
        safe = html.escape(case_id, quote=True)
        body.append(f"<h2>{safe}</h2><a href='{safe}.png'><img loading='lazy' src='{safe}.png' alt='{safe}'></a>")
    (output / "index.html").write_text("\n".join(body), encoding="utf-8")


def save_progress(path, value):
    temporary = path.with_suffix(".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def renderer_signature(study, review, raw, cohort):
    from PIL import __version__ as pillow_version
    from scipy import __version__ as scipy_version
    return {"version": 2, "review_configuration": review_configuration(study),
            "cases_sha256": sha256(review / "cases.csv"), "cohort": cohort, "raw_path": str(raw.resolve()),
            "renderer_sha256": hashlib.sha256(inspect.getsource(make_montage).encode()).hexdigest(),
            "pillow_version": pillow_version, "nibabel_version": nib.__version__,
            "numpy_version": np.__version__, "scipy_version": scipy_version}


def render(args):
    study = json.loads(args.study.read_text())
    validate_review_source(study, args.review)
    labels = panel_labels(study)
    cases = [r for r in read_csv(args.review / "cases.csv") if r["official_split"] == "train"
             and r["cohort"] == args.cohort and r["review_candidate"] == "1"]
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        raise ValueError("No development candidates")
    signature = renderer_signature(study, args.review, args.raw, args.cohort)
    progress_path = args.output / "progress.json"
    if args.output.exists() and not progress_path.exists():
        raise FileExistsError("Existing output without resume provenance")
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"signature": signature, "completed": [], "errors": {}}
    if progress["signature"] != signature:
        raise ValueError("Resume inputs/code changed; use a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    save_progress(progress_path, progress)
    for case in cases:
        case_id = case["case_id"]
        if Path(case_id).name != case_id or case_id in {".", ".."}:
            raise ValueError("Unsafe case ID")
        destination = args.output / (case_id + ".png")
        if case_id in progress["completed"] and destination.exists():
            continue
        start = time.monotonic()
        try:
            montage = make_montage(args.raw, case, labels, study["review_display"])
            temporary = destination.with_suffix(".tmp")
            montage.save(temporary, format="PNG")
            montage.close()
            temporary.replace(destination)
            if case_id not in progress["completed"]:
                progress["completed"].append(case_id)
            progress["errors"].pop(case_id, None)
        except Exception as exc:
            progress["errors"][case_id] = type(exc).__name__ + ": " + str(exc)
        progress["status"] = "rendering"
        save_progress(progress_path, progress)
        write_index(args.output, progress["completed"])
        print(json.dumps({"case_id": case_id, "completed": len(progress["completed"]), "requested": len(cases),
                          "seconds": round(time.monotonic() - start, 1), "error": progress["errors"].get(case_id)}, ensure_ascii=False), flush=True)
    failed = [r["case_id"] for r in cases if r["case_id"] in progress["errors"]]
    progress["status"] = "partial_errors" if failed else "requested_images_complete_manual_review_pending"
    save_progress(progress_path, progress)
    write_index(args.output, progress["completed"])
    print(json.dumps({"status": progress["status"], "index": str(args.output / "index.html")}, ensure_ascii=False))
    if failed:
        raise RuntimeError(f"{len(failed)} cases failed; inspect progress.json and rerun to retry")


def load_review_ledger(study, review, qc):
    """검토 표의 무결성을 status와 cohort 분석에서 동일하게 검사한다."""
    summary = validate_review_source(study, review)
    labels = panel_labels(study)
    cases = {r["case_id"]: r for r in read_csv(review / "cases.csv")
             if r["official_split"] == "train" and r["review_candidate"] == "1"}
    rows = read_csv(review / "review.csv")
    expected = {(case, label) for case in cases for label in labels}
    actual = [(r["case_id"], r["label"]) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Manual review ledger has repeated, missing, or unexpected case/organ rows")
    by_case = collections.defaultdict(list)
    for row in rows:
        if row["cohort"] != cases[row["case_id"]]["cohort"]:
            raise ValueError("Review cohort differs from generated case list")
        for field in ("coverage", "annotation"):
            if row[field] not in summary["manual_review_values"][field]:
                raise ValueError("Invalid manual review value: " + field)
        if (row["coverage"] != "pending" or row["annotation"] != "pending") and (not row["reviewer"].strip() or not row["notes"].strip()):
            raise ValueError("Nonpending judgments require reviewer and notes")
        by_case[row["case_id"]].append(row)
    organ_path = qc / "organ_qc.csv"
    if sha256(organ_path) != summary["provenance"]["organ_qc_sha256"]:
        raise ValueError("Original organ QC changed")
    source_organs = {}
    for row in read_csv(organ_path):
        if row["case_id"] in cases and row["label"] in labels:
            source_organs[(row["case_id"], row["label"])] = row
    for row in rows:
        original = source_organs[(row["case_id"], row["label"])]
        if row["automatic_empty"] != original["empty"] or row["automatic_border"] != original["touches_border"]:
            raise ValueError("Automatic QC columns in manual review ledger were changed")
    return summary, cases, by_case, source_organs


def review_status(args):
    """표와 기존 PNG만 사용한다. 검토 순서를 보여주며 포함·제외 판정을 자동 생성하지 않는다."""
    study = json.loads(args.study.read_text())
    _, cases, by_case, source_organs = load_review_ledger(study, args.review, args.qc)
    smallest = {}
    for (case_id, label), row in source_organs.items():
        if row["empty"] != "0":
            continue
        item = (int(row["nonzero_voxels"]), label)
        if case_id not in smallest or item < smallest[case_id]:
            smallest[case_id] = item
    counts = {}
    for cohort in ("pediatric", "adult"):
        ids = [case for case, row in cases.items() if row["cohort"] == cohort]
        counts[cohort] = {
            "candidates": len(ids),
            "cases_with_any_reviewer_entry": sum(any(r["reviewer"] for r in by_case[case]) for case in ids),
            "annotation_acceptable_rows": sum(r["annotation"] == "acceptable" for case in ids for r in by_case[case]),
            "cases_with_rejected_annotation": sorted(case for case in ids if any(r["annotation"] == "reject" for r in by_case[case]))}
    body = ["<!doctype html><meta charset='utf-8'><title>QC 검토 현황</title>",
            "<style>body{font:16px sans-serif;margin:30px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #bbb}input{padding:8px;width:400px}.hold{background:#ffe0df}</style>",
            "<h1>Development QC 검토 현황</h1><p>이미지는 기존 파일로 연결합니다. 이 표는 검토 우선순위이며 학습 포함 승인표가 아닙니다.</p>",
            "<p>작은 mask를 먼저 보되, voxel 수만으로 삭제하지 않습니다. 경계 접촉이 없어도 촬영 범위가 충분하다는 뜻은 아닙니다.</p>",
            "<p>성인·소아와 case ID로 검색할 수 있습니다. 판정 수정 후 <code>python -m tsv3.review status</code>로 갱신합니다.</p>",
            "<pre>" + html.escape(json.dumps(counts, ensure_ascii=False, indent=2)) + "</pre>",
            "<input placeholder='pediatric, adult 또는 case ID' oninput=\"for(const r of document.querySelectorAll('tbody tr'))r.hidden=!r.textContent.toLowerCase().includes(this.value.toLowerCase())\">",
            "<table><thead><tr><th>cohort</th><th>case / image</th><th>nonempty / 7</th><th>no border / 7</th><th>smallest positive mask (voxels)</th><th>review</th></tr></thead><tbody>"]
    def order(case):
        rejected = any(r["annotation"] == "reject" for r in by_case[case])
        return (not rejected, smallest.get(case, (0, ""))[0], case)
    for case_id in sorted(cases, key=order):
        case = cases[case_id]
        cohort = case["cohort"]
        escaped_id = html.escape(case_id, quote=True)
        image_path = args.output / cohort / (case_id + ".png")
        link = f"<a href='{html.escape(cohort, quote=True)}/{escaped_id}.png'>{escaped_id}</a>" if image_path.exists() else escaped_id + " (image pending)"
        statuses = collections.Counter(r["annotation"] for r in by_case[case_id])
        rejected = statuses["reject"] > 0
        size, label = smallest.get(case_id, (0, "none"))
        body.append(f"<tr class={'hold' if rejected else 'pending'}><td>{html.escape(cohort)}</td><td>{link}</td>"
                    f"<td>{case['nonempty_panel_organs']}</td><td>{case['nonempty_no_border_organs']}</td>"
                    f"<td>{html.escape(label)}: {size}</td><td>{html.escape(str(dict(statuses)))}</td></tr>")
    body.append("</tbody></table>")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "index.html").write_text("\n".join(body), encoding="utf-8")
    print(json.dumps({"status": "review_pending_not_training_ready", "counts": counts,
                      "dashboard": str(args.output / "index.html")}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Fast: reuse QC CSV; never load raw image voxels")
    p.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    p.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    p.add_argument("--qc", type=Path, default=ROOT / "manifests/qc_v300")
    p.add_argument("--output", type=Path, default=DEFAULT_REVIEW)
    p.set_defaults(func=prepare)
    p = sub.add_parser("render", help="Slow: sequential development images; same command resumes")
    p.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    p.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    p.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    p.add_argument("--cohort", choices=("pediatric", "adult"), default="pediatric")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--limit", type=int, default=0, help="Small validation run only; 0 means all candidates")
    p.set_defaults(func=render)
    p = sub.add_parser("status", help="Fast: validate manual ledger and refresh the review dashboard; no image regeneration")
    p.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    p.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    p.add_argument("--qc", type=Path, default=ROOT / "manifests/qc_v300")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/review_v300")
    p.set_defaults(func=review_status)
    args = parser.parse_args()
    if args.command == "render":
        if args.limit < 0:
            parser.error("limit must be >=0")
        if args.output is None:
            args.output = ROOT / "outputs/review_v300" / args.cohort
    args.func(args)


if __name__ == "__main__":
    main()
