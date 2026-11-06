"""공개 reference 기반 test 평가 계약과 공통 Dice·surface metric 구현."""

import argparse
import collections
import importlib.metadata
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from .data import ROOT, DEFAULT_RAW, read_csv, sha256, write_csv, write_json
from .review import panel_labels


def dice_from_counts(intersection, reference_count, prediction_count):
    """Hard Dice. 두 mask가 모두 비면 NaN이며, 호출자가 eligibility로 처리한다."""
    intersection = np.asarray(intersection, dtype=np.float64)
    denominator = np.asarray(reference_count, dtype=np.float64) + prediction_count
    return np.divide(2 * intersection, denominator,
                     out=np.full(np.broadcast_shapes(intersection.shape, np.shape(denominator)), np.nan),
                     where=denominator > 0)


def validation_dice(prediction, target, label_count):
    """Development validation: 합친 target의 overlap(ignore)을 제외한다."""
    if prediction.shape != target.shape:
        raise ValueError("Prediction/target shape differs")
    if prediction.min() < 0 or prediction.max() > label_count:
        raise ValueError("Prediction label outside configured panel")
    valid = target != label_count + 1
    true = target[valid].astype(np.int64)
    pred = prediction[valid].astype(np.int64)
    if not true.size or true.min() < 0 or true.max() > label_count:
        raise ValueError("Invalid target label")
    matrix = np.bincount(true * (label_count + 1) + pred,
                         minlength=(label_count + 1)**2).reshape(label_count + 1, label_count + 1)
    gt = matrix.sum(axis=1)[1:]
    if (gt == 0).any():
        raise ValueError("Development validation requires every positive reference organ")
    return dice_from_counts(matrix.diagonal()[1:], gt, matrix.sum(axis=0)[1:]).tolist()


def score_binary(reference, prediction, tolerance_voxels=2.0):
    """원본 격자의 개별 binary GT를 평가한다. Index-space voxel 단위, mm 가정 없음."""
    from surface_distance import metrics
    if reference.dtype != bool or prediction.dtype != bool or reference.ndim != 3 or reference.shape != prediction.shape:
        raise ValueError("Metrics require matching 3D boolean masks")
    if not np.isfinite(tolerance_voxels) or tolerance_voxels < 0:
        raise ValueError("Surface tolerance must be finite and nonnegative")
    n_ref, n_pred = int(reference.sum()), int(prediction.sum())
    result = {"reference_voxels": n_ref, "prediction_voxels": n_pred,
              "distance_unit": "voxel", "NSD_tolerance_voxels": float(tolerance_voxels)}
    if n_ref == 0:
        return {**result, "status": "empty_reference_unknown_not_scored", "Dice": None,
                "NSD": None, "HD95": None, "HD95_infinite": False}
    dice = float(dice_from_counts(int((reference & prediction).sum()), n_ref, n_pred))
    if n_pred == 0:
        # JSON null + explicit flag represents infinity without nonstandard JSON Infinity.
        return {**result, "status": "empty_prediction", "Dice": 0.0, "NSD": 0.0,
                "HD95": None, "HD95_infinite": True}
    distances = metrics.compute_surface_distances(reference, prediction, spacing_mm=(1.0, 1.0, 1.0))
    nsd = float(metrics.compute_surface_dice_at_tolerance(distances, tolerance_voxels))
    hd95 = float(metrics.compute_robust_hausdorff(distances, 95))
    if not (np.isfinite(dice) and np.isfinite(nsd) and np.isfinite(hd95)):
        raise ValueError("Unexpected nonfinite metric for nonempty masks")
    return {**result, "status": "scored", "Dice": dice, "NSD": nsd, "HD95": hd95, "HD95_infinite": False}


def test_eligibility(split, case_rows, organ_rows, labels):
    """예측값을 받지 않는 사전 규칙. 빈 GT와 border flag를 임상 판정으로 바꾸지 않는다."""
    cases = {row["case_id"]: row for row in case_rows}
    if len(cases) != len(case_rows):
        raise ValueError("Duplicate case QC rows")
    organs = {(row["case_id"], row["label"]): row for row in organ_rows if row["label"] in labels}
    if len(organs) != sum(row["label"] in labels for row in organ_rows):
        raise ValueError("Duplicate organ QC rows")
    records = []
    all_test = split["partitions"]["adult_test"] + split["partitions"]["pediatric_test"]
    if len(set(all_test)) != len(all_test):
        raise ValueError("Test cohort overlap")
    development = {c for role, ids in split["partitions"].items() if role not in {"adult_test", "pediatric_test", "excluded"} for c in ids}
    if development.intersection(all_test):
        raise ValueError("Development case entered test evaluation")
    for cohort in ("adult", "pediatric"):
        for case in sorted(split["partitions"][cohort + "_test"]):
            row = cases[case]
            if row["official_split"] != "test" or row["cohort"] != cohort or row["automated_eligible"] != "1":
                raise ValueError("Frozen test membership/QC mismatch: " + case)
            for label in labels:
                organ = organs[case, label]
                if organ["error"] or organ["empty"] not in {"0", "1"} or organ["touches_border"] not in {"0", "1"}:
                    raise ValueError("Invalid technical QC in frozen test: " + case + "/" + label)
                positive = int(organ["nonzero_voxels"]) > 0
                if positive != (organ["empty"] == "0"):
                    raise ValueError("Inconsistent mask QC counts")
                interior = positive and organ["touches_border"] == "0"
                records.append({"case_id": case, "cohort": cohort, "label": label,
                                "reference_voxels": int(organ["nonzero_voxels"]),
                                "touches_border": int(organ["touches_border"]),
                                "primary_eligible": int(interior), "sensitivity_eligible": int(positive),
                                "primary_reason": "public_reference_nonempty_no_border" if interior else
                                    ("empty_reference_cause_unknown" if not positive else "reference_touches_image_border"),
                                "manual_full_volume_review": "not_performed"})
    return records


def metric_identity():
    from surface_distance import metrics, lookup_tables
    return {"package": "surface-distance", "version": importlib.metadata.version("surface-distance"),
            "metrics_sha256": sha256(metrics.__file__), "lookup_tables_sha256": sha256(lookup_tables.__file__),
            "numpy": np.__version__, "scipy": importlib.metadata.version("scipy")}


def evaluation_function_identity():
    """사전 정의의 의미를 고정한다. 관련 없는 CLI 확장이 재고정을 요구하지 않게 한다."""
    return {function.__name__: hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
            for function in (dice_from_counts, validation_dice, score_binary, test_eligibility)}


def freeze_evaluation(args):
    from .preprocessing import load_plan
    plan = load_plan(args)
    study, split = plan["study"], plan["split"]
    evaluation = study["evaluation"]
    if evaluation["distance_unit"] != "voxel" or evaluation["NSD_tolerance_voxels"] != 2:
        raise ValueError("Only the declared voxel-space metric contract is supported")
    if evaluation["organ_eligibility"] != "technical_QC_public_nonempty_no_border_primary_all_nonempty_sensitivity":
        raise ValueError("Unsupported test eligibility policy")
    if args.output.exists():
        raise FileExistsError("Evaluation contract exists; keep it immutable")
    for name, key in (("case_qc.csv", "case_qc_sha256"), ("organ_qc.csv", "organ_qc_sha256")):
        if sha256(args.qc / name) != split["provenance"][key]:
            raise ValueError("Evaluation QC differs from frozen split inputs")
    case_rows = read_csv(args.qc / "case_qc.csv")
    organ_rows = read_csv(args.qc / "organ_qc.csv")
    records = test_eligibility(split, case_rows, organ_rows, plan["labels"])
    by_case = {row["case_id"]: row for row in case_rows}
    counts = {cohort: {label: {policy: sum(int(r[policy + "_eligible"]) for r in records
                         if r["cohort"] == cohort and r["label"] == label)
                         for policy in ("primary", "sensitivity")} for label in plan["labels"]}
              for cohort in ("adult", "pediatric")}
    if any(v["primary"] == 0 for cohort in counts.values() for v in cohort.values()):
        raise ValueError("An organ/cohort has no primary eligible reference")
    contract = {"status": "test_evaluation_frozen_before_research_training", "labels": study["labels"],
                "evaluation": evaluation, "split_sha256": sha256(ROOT / study["cohort"]["split_manifest"]),
                "qc_sha256": {name: sha256(args.qc / name) for name in ("case_qc.csv", "organ_qc.csv")},
                "metric_identity": metric_identity(), "implementation_functions": evaluation_function_identity(),
                "test_case_ids": {c: split["partitions"][c + "_test"] for c in ("adult", "pediatric")},
                "counts": counts, "raw_sources": {}, "case_CT_content_sha256": {},
                "limitations": ["Public reference masks; no expert full-volume test review.",
                                "No-border screening does not establish full organ coverage or annotation correctness.",
                                "Different organs have different case subsets; heterogeneity may reflect composition.",
                                "Primary pediatric organ samples are small; intervals are exploratory.",
                                "Distances use voxel index space, not verified physical millimetres."]}
    # A single streaming pass freezes source identity; no image rendering or model predictions.
    for case in sorted({r["case_id"] for r in records}):
        contract["case_CT_content_sha256"][case] = by_case[case]["ct_voxel_sha256"]
        paths = [args.raw / case / "ct.nii.gz", *[args.raw / case / "segmentations" / (label + ".nii.gz") for label in plan["labels"]]]
        for path in paths:
            contract["raw_sources"][str(path.relative_to(args.raw))] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    common = {}
    for cohort in ("adult", "pediatric"):
        counter = collections.Counter(r["case_id"] for r in records if r["cohort"] == cohort and r["primary_eligible"])
        common[cohort] = sorted(c for c, n in counter.items() if n == len(plan["labels"]))
    contract["all_7_primary_eligible_cases_for_descriptive_sensitivity"] = common
    args.output.mkdir(parents=True)
    write_csv(args.output / "eligibility.csv", records, list(records[0]))
    contract["eligibility_sha256"] = sha256(args.output / "eligibility.csv")
    write_json(args.output / "contract.json", contract)
    print(json.dumps({"status": contract["status"], "counts": counts, "output": str(args.output)}, ensure_ascii=False, indent=2))


def validate_evaluation_contract(study):
    """학습 시작 전 manifest·계산 정의·입력 identity를 확인한다. CT 재읽기는 하지 않는다."""
    path = ROOT / study["evaluation"]["manifest"]
    contract = json.loads(path.read_text())
    if contract["status"] != "test_evaluation_frozen_before_research_training" or contract["labels"] != study["labels"]:
        raise ValueError("Evaluation contract is incomplete or has different labels")
    if contract["evaluation"] != study["evaluation"]:
        raise ValueError("Evaluation settings changed after freeze")
    if contract["split_sha256"] != sha256(ROOT / study["cohort"]["split_manifest"]):
        raise ValueError("Split changed after evaluation freeze")
    if contract["implementation_functions"] != evaluation_function_identity() or contract["metric_identity"] != metric_identity():
        raise ValueError("Metric implementation changed after freeze")
    for name, digest in contract["qc_sha256"].items():
        if sha256(ROOT / "manifests/qc_v300" / name) != digest:
            raise ValueError("QC changed after evaluation freeze")
    if sha256(path.parent / "eligibility.csv") != contract["eligibility_sha256"]:
        raise ValueError("Frozen test eligibility changed")
    return {"contract_sha256": sha256(path), "eligibility_sha256": contract["eligibility_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true", required=True)
    parser.add_argument("--study", type=Path, default=ROOT / "configs/study.json")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--review", type=Path, default=ROOT / "manifests/review_v300")
    parser.add_argument("--qc", type=Path, default=ROOT / "manifests/qc_v300")
    parser.add_argument("--output", type=Path, default=ROOT / "manifests/evaluation_abdominal_v1")
    freeze_evaluation(parser.parse_args())


if __name__ == "__main__":
    main()
