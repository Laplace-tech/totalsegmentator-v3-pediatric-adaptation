"""공개 reference 기반 test 평가 계약과 공통 Dice·surface metric 구현."""

import argparse
import collections
import importlib.metadata
import hashlib
import inspect
import json
import sqlite3
import time
from contextlib import closing
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


MODELS = ("M0", "M1", "M2")
METRICS = ("Dice", "NSD", "HD95")


def verified_input(raw, relative, sources):
    path = raw / relative
    record = sources[relative]
    if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
        raise ValueError(f"Raw input differs from the frozen record: {relative}")
    return path


def prepare_test_case(raw, case, labels, sources, normalization):
    import nibabel as nib
    from .audit import geometry, geometry_delta_voxels
    from .preprocessing import normalize_inplace

    source = nib.load(verified_input(raw, f"{case}/ct.nii.gz", sources))
    shape, spacing = geometry(source)
    canonical = nib.as_closest_canonical(source)
    data = np.asarray(canonical.dataobj, dtype=np.float32).copy()[None]
    if not np.isfinite(data).all():
        raise ValueError(f"Nonfinite CT: {case}")
    normalize_inplace(data, normalization)
    inverse = nib.orientations.ornt_transform(nib.orientations.io_orientation(canonical.affine),
                                              nib.orientations.io_orientation(source.affine))
    restored_affine = canonical.affine @ nib.orientations.inv_ornt_aff(inverse, data.shape[1:])
    if not np.allclose(restored_affine, source.affine, rtol=0, atol=1e-6):
        raise ValueError(f"Inverse orientation changed the source grid: {case}")

    references = {}
    for label in labels:
        relative = f"{case}/segmentations/{label}.nii.gz"
        mask = nib.load(verified_input(raw, relative, sources))
        if (mask.shape != shape or mask.header.get_xyzt_units()[0] != source.header.get_xyzt_units()[0]
                or geometry_delta_voxels(mask.affine, source.affine, shape, spacing) > 0.1):
            raise ValueError(f"Reference grid differs from CT: {relative}")
        values = np.asanyarray(mask.dataobj)
        if not np.logical_or(values == 0, values == 1).all():
            raise ValueError(f"Nonbinary reference: {relative}")
        references[label] = values.astype(bool, copy=False)
    return data, inverse, shape, references


def score_prepared_case(model, prepared, labels, config, tolerance):
    from nibabel.orientations import apply_orientation
    from .training import infer_volume

    data, inverse, shape, references = prepared
    started = time.monotonic()
    prediction, timing = infer_volume(model, data, labels, config)
    prediction = apply_orientation(prediction, inverse)
    if prediction.shape != shape:
        raise ValueError("Restored prediction shape differs from original CT")
    scores = {label: score_binary(references[label], prediction == index, tolerance)
              for index, label in enumerate(labels, 1)}
    return {"organs": scores, "inference_seconds": timing["seconds"],
            "elapsed_seconds": time.monotonic() - started, "shape": list(shape),
            "peak_reserved_mib": timing["peak_reserved_mib"]}


def open_score_store(db, identity):
    db.execute("CREATE TABLE IF NOT EXISTS run (id INTEGER PRIMARY KEY CHECK(id=1), identity TEXT NOT NULL)")
    db.execute("""CREATE TABLE IF NOT EXISTS scores (
        seed INTEGER NOT NULL, model TEXT NOT NULL, case_id TEXT NOT NULL, result TEXT NOT NULL,
        PRIMARY KEY(seed, model, case_id))""")
    signature = json.dumps(identity, sort_keys=True, allow_nan=False)
    saved = db.execute("SELECT identity FROM run WHERE id=1").fetchone()
    if saved is None:
        if db.execute("SELECT 1 FROM scores LIMIT 1").fetchone():
            raise ValueError("Existing scores lack run identity")
        with db:
            db.execute("INSERT INTO run VALUES (1, ?)", (signature,))
    elif saved[0] != signature:
        raise ValueError("Evaluation inputs, code or runtime changed since the saved run")


def store_score(db, key, result):
    with db:
        db.execute("INSERT INTO scores VALUES (?, ?, ?, ?)", (*key, json.dumps(result, allow_nan=False)))


def evaluation_inputs(args):
    from .night import verify_completion
    from .preprocessing import load_plan
    from .training import branch_config

    plan = load_plan(args)
    study, labels = plan["study"], plan["labels"]
    cache = ROOT / "data/preprocessed/abdominal_v1"
    manifest, norm = verify_completion(cache, plan)
    evaluation_id = validate_evaluation_contract(study)
    contract_path = ROOT / study["evaluation"]["manifest"]
    contract = json.loads(contract_path.read_text())
    cohorts = contract["test_case_ids"]
    if any(sorted(ids) != sorted(plan["split"]["partitions"][cohort + "_test"])
           for cohort, ids in cohorts.items()):
        raise ValueError("Contract test membership differs from split")
    if set(plan["roles"]).intersection(case for ids in cohorts.values() for case in ids):
        raise ValueError("Development and test inputs intersect")

    seeds = study["training"]["replication_plan"]["seeds"]
    aggregate_path = ROOT / f"outputs/training_{study['study_version']}_seed{seeds[-1]}/replication_validation.json"
    development = json.loads(aggregate_path.read_text())
    common = development["common_input_identity"]
    if (development["seeds"] != seeds or development["planned_seeds"] != seeds
            or common != {"split_sha256": plan["signature"]["split_sha256"],
                          "normalization_sha256": manifest["normalization"]["sha256"],
                          "evaluation_contract_sha256": evaluation_id["contract_sha256"],
                          "eligibility_sha256": evaluation_id["eligibility_sha256"]}):
        raise ValueError("Expected verified all-seed development report with matching inputs")

    checkpoints = {}
    for seed in seeds:
        directory = ROOT / f"outputs/training_{study['study_version']}_seed{seed}"
        run_path = directory / "run.json"
        run = json.loads(run_path.read_text())
        files = development["input_file_sha256"]
        if run["status"] != "complete" or sha256(run_path) != files[str(run_path.relative_to(ROOT))]:
            raise ValueError(f"Run differs from the verified development report: {seed}")
        integrity = development["integrity"][str(seed)]
        if not integrity["M1_M2_same_M0_selected_initialization_and_numeric_config"]:
            raise ValueError(f"M1/M2 checkpoint pairing not verified: {seed}")
        for model in MODELS:
            status_path = directory / model / "status.json"
            if sha256(status_path) != files[str(status_path.relative_to(ROOT))]:
                raise ValueError(f"Branch status differs from the verified report: {seed}/{model}")
            status = json.loads(status_path.read_text())
            checkpoint = directory / model / "selected.pt"
            config = branch_config(run["study_snapshot"], model, False, seed)
            budget = config["optimizer_updates"]
            if (status["status"] != "complete" or status["updates"] != budget
                    or status["selected_checkpoint_sha256"] != integrity["selected_sha256"][model]
                    or sha256(checkpoint) != status["selected_checkpoint_sha256"]
                    or config != branch_config(study, model, False, seed)):
                raise ValueError(f"Checkpoint/status/config mismatch: {seed}/{model}")
            checkpoints[seed, model] = (checkpoint, config, status["selected_checkpoint_sha256"])

    rows = read_csv(contract_path.parent / "eligibility.csv")
    expected = {(cohort, case, label) for cohort, cases in cohorts.items() for case in cases for label in labels}
    keys = {(r["cohort"], r["case_id"], r["label"]) for r in rows}
    if keys != expected or len(rows) != len(expected):
        raise ValueError("Incomplete or duplicated eligibility records")
    identity = {**common, "development_report_sha256": sha256(aggregate_path),
                "checkpoints_sha256": {f"{s}/{m}": value[2] for (s, m), value in checkpoints.items()},
                "code_sha256": {name: sha256(Path(__file__).with_name(name)) for name in
                                ("evaluation.py", "training.py", "preprocessing.py", "audit.py")},
                "uv_lock_sha256": sha256(ROOT / "uv.lock")}
    return plan, manifest, norm, contract, rows, checkpoints, identity


def summarize_delta(left, right, uncertainty, confidence=True):
    """Rows are training seeds, columns are the same paired cases."""
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    delta = left - right
    seed_means = [float(row[mask].mean()) if mask.any() else None for row, mask in zip(delta, valid)]
    common = valid.all(axis=0)
    paired = delta[:, common]
    by_case = paired.mean(axis=0)
    per_seed_common = paired.mean(axis=1) if common.any() else None
    interval = None
    if confidence and len(by_case) >= 2:
        rng = np.random.default_rng(uncertainty["seed"])
        indices = rng.integers(0, len(by_case), size=(uncertainty["bootstrap_replicates"], len(by_case)))
        alpha = (1 - uncertainty["confidence_level"]) / 2
        interval = np.quantile(by_case[indices].mean(axis=1), [alpha, 1 - alpha]).tolist()
    return {"eligible_cases": int(left.shape[1]), "all_seed_finite_paired_cases": int(common.sum()),
            "per_seed_mean": seed_means, "per_seed_finite_pairs": valid.sum(axis=1).tolist(),
            "mean": float(by_case.mean()) if common.any() else None,
            "seed_sample_SD": float(per_seed_common.std(ddof=1)) if common.any() and left.shape[0] > 1 else None,
            "positive_seed_count": sum(value is not None and value > 0 for value in seed_means),
            "negative_seed_count": sum(value is not None and value < 0 for value in seed_means),
            "pointwise_paired_case_interval": interval}


def summarize_official_scores(results, seeds, labels, contract, rows):
    from scipy.stats import spearmanr

    uncertainty = contract["evaluation"]["uncertainty"]
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["cohort"], row["label"]].append(row)
    summary = {}
    analysis = {}
    for cohort in contract["test_case_ids"]:
        summary[cohort] = {}
        common_cases = set(contract["all_7_primary_eligible_cases_for_descriptive_sensitivity"][cohort])
        for policy in ("primary", "nonempty_sensitivity", "all_7_primary_sensitivity"):
            by_organ = summary[cohort][policy] = {}
            for label in labels:
                records = groups[cohort, label]
                cases = [row["case_id"] for row in records if
                         (row["case_id"] in common_cases if policy == "all_7_primary_sensitivity" else
                          row["primary_eligible" if policy == "primary" else "sensitivity_eligible"] == "1")]
                empty_cases = [row["case_id"] for row in records if row["reference_voxels"] == "0"]
                scores = {m: [[results[s, m, c]["organs"][label] for c in cases] for s in seeds] for m in MODELS}
                organ = by_organ[label] = {"eligible_cases": len(cases), "empty_reference_cases": len(empty_cases),
                                          "models": {}, "contrasts": {}}
                for model in MODELS:
                    model_summary = organ["models"][model] = {}
                    for metric in METRICS:
                        values = [[r[metric] for r in group if r[metric] is not None] for group in scores[model]]
                        means = [float(np.mean(v)) if v else None for v in values]
                        model_summary[metric] = {"per_seed_mean": means,
                                                 "mean_of_seed_means": float(np.mean(means)) if all(m is not None for m in means) else None,
                                                 "per_seed_finite_cases": [len(v) for v in values]}
                    model_summary["per_seed_empty_predictions"] = [sum(r["HD95_infinite"] for r in group)
                                                                   for group in scores[model]]
                    if policy == "primary":
                        model_summary["empty_reference_prediction_voxels"] = [
                            {c: results[s, model, c]["organs"][label]["prediction_voxels"] for c in empty_cases}
                            for s in seeds]
                for left, right in (("M1", "M2"), ("M1", "M0"), ("M2", "M0")):
                    contrast = organ["contrasts"][f"{left}_minus_{right}"] = {}
                    for metric in METRICS:
                        a, b = ([[r[metric] for r in group] for group in scores[m]] for m in (left, right))
                        contrast[metric] = summarize_delta(a, b, uncertainty, confidence=policy == "primary")
                    contrast["HD95_failure_transitions_per_seed"] = [
                        {"failure_to_finite": sum(b["HD95_infinite"] and not a["HD95_infinite"] for a, b in pairs),
                         "finite_to_failure": sum(a["HD95_infinite"] and not b["HD95_infinite"] for a, b in pairs),
                         "both_failed": sum(a["HD95_infinite"] and b["HD95_infinite"] for a, b in pairs)}
                        for pairs in [list(zip(a, b)) for a, b in zip(scores[left], scores[right])]]
                if cohort == "pediatric" and policy == "primary":
                    analysis[label] = {"mean_GT_voxels": float(np.mean([int(row["reference_voxels"]) for row in records
                                                                     if row["case_id"] in cases])),
                                       "M0_Dice": organ["models"]["M0"]["Dice"]["mean_of_seed_means"],
                                       "M1_minus_M0_Dice": organ["contrasts"]["M1_minus_M0"]["Dice"]["mean"]}
    gain = [v["M1_minus_M0_Dice"] for v in analysis.values()]
    associations = {}
    for name in ("mean_GT_voxels", "M0_Dice"):
        values = [v[name] for v in analysis.values()]
        rho = spearmanr(values, gain).statistic if len(set(values)) > 1 and len(set(gain)) > 1 else None
        associations[name] = None if rho is None else float(rho)
    return {"results": summary, "pediatric_exploratory_analysis": {"per_organ": analysis, "Spearman_rho": associations,
            "limitations": ["Seven organs, left and right organs are related, eligible case mix varies by organ.",
                            "The baseline and the gain relative to the baseline are mathematically coupled."]}}


def write_test_report(path, report):
    def effect(value):
        mean, sd, interval = value["mean"], value["seed_sample_SD"], value["pointwise_paired_case_interval"]
        if mean is None:
            return "NA"
        span = f" [{interval[0]:+.4f}, {interval[1]:+.4f}]" if interval is not None else ""
        return f"{mean:+.4f} ± {sd:.4f}{span} (n={value['all_seed_finite_paired_cases']})"

    lines = ["# 공식 test 결과", "", f"평가 모델 {report['models']}개, 모델–case 평가 {report['jobs']}건.",
             "원본 voxel grid의 Dice, NSD(2 voxel), HD95(voxel). Δ는 표의 왼쪽 모델에서 오른쪽 모델을 뺀 값이다.",
             "평균 ± seed 표본 SD [case bootstrap 95% 범위]. HD95는 모든 seed에서 두 모델 모두 유한한 경우의 조건부 효과다."]
    for cohort, comparisons in (("pediatric", ("M1_minus_M2", "M1_minus_M0")),
                                ("adult", ("M1_minus_M0", "M2_minus_M0"))):
        for comparison in comparisons:
            lines += ["", f"## {cohort} {comparison}", "", "| 장기 | 적격 case | ΔDice | ΔNSD | ΔHD95 |",
                      "|---|---:|---|---|---|"]
            for label, organ in report["results"][cohort]["primary"].items():
                cells = [label, str(organ["eligible_cases"])] + [effect(organ["contrasts"][comparison][m]) for m in METRICS]
                lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "장기별 적격 case가 다르며 소아 주 분석은 장기당 6–9건이다. 동일 case를 seed마다 재사용했다.",
              "HD95 실패 전이, 민감도 분석, seed별 수치 및 탐색 분석은 `summary.json`에 기록했다.", ""]
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def run_official_evaluation(args):
    import fcntl
    import torch
    from .preprocessing import atomic_json, check_space
    from .training import build_model

    plan, manifest, normalization, contract, rows, checkpoints, identity = evaluation_inputs(args)
    labels = plan["labels"]
    counts = contract["counts"]
    cohort_cases = contract["test_case_ids"]
    cases = sorted(case for ids in cohort_cases.values() for case in ids)
    for relative, record in contract["raw_sources"].items():
        if (args.raw / relative).stat().st_size != record["bytes"]:
            raise ValueError(f"Frozen source size changed: {relative}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the fixed model inference")
    identity["runtime"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                           "gpu": torch.cuda.get_device_name()}
    first_path, config, _ = next(iter(checkpoints.values()))
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    tolerance = contract["evaluation"]["NSD_tolerance_voxels"]
    expected = {(seed, model, case) for seed, model in checkpoints for case in cases}
    if args.preflight:
        print(json.dumps({"status": "official_test_preflight_passed", "checkpoints": len(checkpoints),
                          "jobs": len(expected), "cases": {c: len(ids) for c, ids in cohort_cases.items()},
                          "counts": counts, "identity": identity}, indent=2), flush=True)
        return
    if args.smoke:
        model = build_model(config, len(labels) + 1)
        payload = torch.load(first_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        del payload
        smoke_results = []
        with closing(sqlite3.connect(":memory:")) as db:
            open_score_store(db, {"development_smoke": True})
            for role in ("adult_validation", "pediatric_validation"):
                ids = plan["split"]["partitions"][role]
                case = sorted(ids, key=lambda c: np.prod(manifest["completed"][c]["properties"]["source_shape"]))[len(ids) // 2]
                sources = manifest["completed"][case]["source_files"]
                prepared = prepare_test_case(args.raw, case, labels, sources, normalization)
                sample = score_prepared_case(model, prepared, labels, config, tolerance)
                key = (config["seed"], "M0", case)
                store_score(db, key, sample)
                saved = db.execute("SELECT result FROM scores WHERE seed=? AND model=? AND case_id=?", key).fetchone()
                if json.loads(saved[0]) != sample:
                    raise ValueError("Smoke score did not round trip")
                smoke_results.append({"cohort": role, "case": case, **{k: sample[k] for k in
                                     ("elapsed_seconds", "inference_seconds", "peak_reserved_mib", "shape")}})
                del prepared
        print(json.dumps({"status": "development_smoke_passed", "timings": smoke_results}, indent=2), flush=True)
        return

    output = args.output or ROOT / f"outputs/test_{plan['study']['study_version']}"
    output.mkdir(parents=True, exist_ok=True)
    check_space(output, args.host_volume, 64 * 2**20, config["disk_reserve_gib"])
    with (output / ".lock").open("a") as lock, closing(sqlite3.connect(output / "scores.sqlite")) as db:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        open_score_store(db, identity)
        done = {(s, m, c) for s, m, c in db.execute("SELECT seed, model, case_id FROM scores")}
        if not done <= expected:
            raise ValueError("Saved scores include unexpected cases or models")
        print(json.dumps({"status": "running", "completed": len(done), "total": len(expected)}), flush=True)
        if done != expected:
            weights = {}
            for (seed, branch), (path, branch_settings, _) in checkpoints.items():
                payload = torch.load(path, map_location="cpu", weights_only=False)
                if payload["config"] != branch_settings:
                    raise ValueError(f"Checkpoint config differs: {seed}/{branch}")
                weights[seed, branch] = payload["model"]
                del payload
            model = build_model(config, len(labels) + 1)
            refs = {(r["case_id"], r["label"]): int(r["reference_voxels"]) for r in rows}
            for case in cases:
                pending = [(seed, branch) for seed, branch in checkpoints if (seed, branch, case) not in done]
                if not pending:
                    continue
                prepared = prepare_test_case(args.raw, case, labels, contract["raw_sources"], normalization)
                for seed, branch in pending:
                    model.load_state_dict(weights[seed, branch])
                    result = score_prepared_case(model, prepared, labels, checkpoints[seed, branch][1], tolerance)
                    if any(r["reference_voxels"] != refs[case, label] for label, r in result["organs"].items()):
                        raise ValueError(f"Reference volume differs from frozen eligibility: {case}")
                    key = (seed, branch, case)
                    store_score(db, key, result)
                    done.add(key)
                    print(json.dumps({"completed": len(done), "total": len(expected), "case": case, "seed": seed,
                                      "model": branch, "seconds": round(result["elapsed_seconds"], 2)}), flush=True)
                del prepared
            del weights, model
        if done != expected:
            raise ValueError("Incomplete score store")
        results = {(s, m, c): json.loads(value) for s, m, c, value in
                   db.execute("SELECT seed, model, case_id, result FROM scores")}
        seeds = list(dict.fromkeys(s for s, _ in checkpoints))
        report = {"status": "official_test_complete", "models": len(checkpoints), "jobs": len(results),
                  "seeds": seeds, "identity": identity, "limitations": contract["limitations"],
                  **summarize_official_scores(results, seeds, labels, contract, rows)}
        atomic_json(output / "summary.json", report)
        write_test_report(output / "report.md", report)
        print(json.dumps({"status": report["status"], "output": str(output), "jobs": len(results)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--preflight", action="store_true", help="Verify frozen inputs and selected models without inference")
    mode.add_argument("--smoke", action="store_true", help="Run the test scoring path on two development cases")
    mode.add_argument("--official-test", action="store_true", help="Score the frozen test and resume from committed scores")
    parser.add_argument("--study", type=Path, default=ROOT / "configs/study.json")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--review", type=Path, default=ROOT / "manifests/review_v300")
    parser.add_argument("--qc", type=Path, default=ROOT / "manifests/qc_v300")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--host-volume", type=Path)
    args = parser.parse_args()
    if args.freeze:
        args.output = args.output or ROOT / "manifests/evaluation_abdominal_v1"
        freeze_evaluation(args)
    else:
        run_official_evaluation(args)


if __name__ == "__main__":
    main()
