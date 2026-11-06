"""기존 QC 임시 split을 재현하고 현재 연구의 development cohort·K를 검토한다."""

import argparse
import collections
import datetime
import json
import random
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from .data import ROOT, DEFAULT_RAW, DEFAULT_INVENTORY, DEFAULT_PROTOCOL, load_inputs, read_csv, sha256, write_csv, write_json, git_provenance


def allocation_counts(cohort_sizes, k, validation):
    """동일 K·validation 예산의 산술적 배정 가능성. 표본의 통계적 충분성을 뜻하지 않는다."""
    if any(type(value) is not int or value < 1 for value in (k, validation)):
        raise ValueError("K and validation size must be positive integers")
    adult, pediatric = cohort_sizes["adult"], cohort_sizes["pediatric"]
    if adult < k + validation + 1 or pediatric < k + validation:
        return None
    return {"M0_train": adult - k - validation, "M1_finetune": k, "M2_finetune": k,
            "adult_validation": validation, "pediatric_validation": validation,
            "unused_pediatric": pediatric - k - validation}


def make_splits(rows, settings):
    ids = [r["case_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Repeated case IDs in QC table")
    rng = random.Random(settings["seed"])
    k, v = settings["finetuning_cases_per_arm"], settings["validation_cases_per_cohort"]
    train_by_cohort = {cohort: sorted(r["case_id"] for r in rows if r["cohort"] == cohort
                                    and r["official_split"] == "train" and int(r["automated_eligible"]) == 1)
                       for cohort in ("adult", "pediatric")}
    if allocation_counts({cohort: len(ids) for cohort, ids in train_by_cohort.items()}, k, v) is None:
        raise ValueError(f"Insufficient development cases for K={k}, validation={v}")
    pools = {}
    for cohort in ("adult", "pediatric"):
        train = train_by_cohort[cohort]
        rng.shuffle(train)
        pools[cohort + "_validation"] = sorted(train[:v])
        pools["M2_finetune" if cohort == "adult" else "M1_finetune"] = sorted(train[v:v+k])
        pools["M0_train" if cohort == "adult" else "unused_pediatric"] = sorted(train[v+k:])
        pools[cohort + "_test"] = sorted(r["case_id"] for r in rows if r["cohort"] == cohort
                                          and r["official_split"] == "test" and int(r["automated_eligible"]) == 1)
    pools["excluded"] = sorted(r["case_id"] for r in rows if int(r["automated_eligible"]) != 1)
    validate_splits(pools, rows, settings)
    return pools


def validate_splits(pools, rows, settings):
    by_id = {r["case_id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("Repeated case IDs in QC table")
    role_cohorts = {"M0_train": "adult", "M2_finetune": "adult", "adult_validation": "adult", "adult_test": "adult",
                    "M1_finetune": "pediatric", "pediatric_validation": "pediatric", "pediatric_test": "pediatric",
                    "unused_pediatric": "pediatric"}
    if set(pools) != set(role_cohorts) | {"excluded"}:
        raise ValueError("Unexpected or missing split roles")
    assigned = [x for group in pools.values() for x in group]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(by_id):
        raise ValueError("Split overlap or unassigned cases")
    seen_hash = {}
    for name, cases in pools.items():
        for case_id in cases:
            row = by_id[case_id]
            if name == "excluded":
                if int(row["automated_eligible"]) == 1:
                    raise ValueError("Eligible case silently excluded")
                continue
            if row["cohort"] != role_cohorts[name] or int(row["automated_eligible"]) != 1:
                raise ValueError("Ineligible case or wrong cohort in " + name)
            is_test = name.endswith("_test")
            if (row["official_split"] == "test") != is_test:
                raise ValueError("Official test membership changed")
            if not is_test:
                if case_id in settings["manual_training_exclusions"]:
                    raise ValueError("Quarantined case entered development data")
            h = row["ct_voxel_sha256"]
            if not h:
                raise ValueError("Missing CT content hash")
            if h in seen_hash:
                raise ValueError("Repeated CT content among retained cases")
            seen_hash[h] = case_id
    for name in ("M1_finetune", "M2_finetune"):
        if len(pools[name]) != settings["finetuning_cases_per_arm"]:
            raise ValueError("Unequal fine-tuning case budget")
    for name in ("adult_validation", "pediatric_validation"):
        if len(pools[name]) != settings["validation_cases_per_cohort"]:
            raise ValueError("Incorrect validation case budget")
    if not pools["M0_train"]:
        raise ValueError("M0 training cohort is empty")


def study_eligibility(rows, cases, reviews, label_count):
    """공개 reference 기반 development screen. 수동 승인 상태나 test eligibility는 바꾸지 않는다."""
    screened = []
    for original in rows:
        row = dict(original)
        reasons = []
        if row["official_split"] == "train" and int(row["automated_eligible"]) == 1:
            case = cases.get(row["case_id"])
            if case is None or int(case["nonempty_no_border_organs"]) != label_count:
                reasons.append("development_screen_requires_all_panel_nonempty_no_border")
            if case is not None:
                entries = reviews[row["case_id"]]
                if len(entries) != label_count:
                    raise ValueError("Incomplete review ledger")
                if any(r["annotation"] == "reject" for r in entries):
                    reasons.append("recorded_annotation_rejection")
                if any(r["coverage"] in {"partial", "out_of_fov", "true_absence"} for r in entries):
                    reasons.append("recorded_incomplete_panel_coverage")
            if reasons:
                row["automated_eligible"] = "0"
                row["exclusion_reasons"] = ";".join(filter(None, [row["exclusion_reasons"], *reasons]))
        screened.append(row)
    return screened


def freeze_development(args):
    from .review import load_review_ledger, panel_labels
    study = json.loads(args.study.read_text())
    cohort = study["cohort"]
    if cohort.get("reference_label_policy") != "public_reference_with_technical_QC_and_recorded_manual_holds":
        raise ValueError("Public-reference label policy has not been adopted")
    if cohort["training_candidate_screen"] != "all_panel_nonempty_no_border":
        raise ValueError("Unsupported development screen")
    summary, cases, reviews, _ = load_review_ledger(study, args.review, args.qc)
    _, metadata, _, protocol, provenance = load_inputs(args.raw, DEFAULT_INVENTORY, ROOT / study["qc_protocol"])
    for key, value in provenance.items():
        if summary["provenance"].get(key) != value:
            raise ValueError("Split input provenance mismatch: " + key)
    qc_summary = json.loads((args.qc / "summary.json").read_text())
    if qc_summary["completed_cases"] != qc_summary["expected_cases"] or qc_summary["status"] == "partial":
        raise ValueError("Full automated QC must complete before splitting")
    if sha256(args.qc / "case_qc.csv") != summary["provenance"]["case_qc_sha256"]:
        raise ValueError("Case QC source changed")
    settings = {**protocol, "seed": cohort["split_seed"],
                "finetuning_cases_per_arm": cohort["finetuning_cases_per_arm"],
                "validation_cases_per_cohort": cohort["validation_cases_per_cohort"]}
    rows = study_eligibility(read_csv(args.qc / "case_qc.csv"), cases, reviews, len(panel_labels(study)))
    pools = make_splits(rows, settings)
    development_roles = ("M0_train", "M1_finetune", "M2_finetune", "adult_validation", "pediatric_validation")
    covariates = {}
    for role in development_roles:
        records = [metadata[case] for case in pools[role]]
        covariates[role] = {"age": numeric_summary([r["age"] for r in records]),
                            "source_slice_thickness": numeric_summary([r["slice_thickness"] for r in records], positive_only=True),
                            **{key: dict(collections.Counter(r[key].strip() or "<missing>" for r in records))
                               for key in ("institute", "study_type", "gender")}}
    provenance.update({"study_sha256": sha256(args.study), "case_qc_sha256": sha256(args.qc / "case_qc.csv"),
                       "organ_qc_sha256": sha256(args.qc / "organ_qc.csv"),
                       "review_csv_sha256": sha256(args.review / "review.csv"),
                       "review_cases_sha256": sha256(args.review / "cases.csv"),
                       "split_code_sha256": sha256(__file__), **git_provenance()})
    result = {"status": "development_split_frozen_test_organ_eligibility_pending", "seed": settings["seed"],
              "unit": cohort["split_unit"], "label_policy": cohort["reference_label_policy"],
              "study_snapshot": study, "provenance": provenance,
              "counts": {role: len(ids) for role, ids in pools.items()}, "partitions": pools,
              "development_covariates": covariates,
              "limitations": ["Public reference labels; no comprehensive expert full-volume review.",
                              "Nonempty/no-border screening does not prove full organ coverage or annotation accuracy.",
                              "Development screening selects a narrower cohort; generalization is limited.",
                              "Case independence is not verified patient independence.",
                              "Official test pools are preserved; case-organ scoring eligibility remains pending."]}
    if args.output.exists():
        raise FileExistsError("Split outputs are immutable; choose a new version directory")
    args.output.mkdir(parents=True)
    write_json(args.output / "splits.json", result)
    assignments = {case: role for role, ids in pools.items() for case in ids}
    write_csv(args.output / "assignments.csv",
              ({"case_id": r["case_id"], "partition": assignments[r["case_id"]], "cohort": r["cohort"],
                "exclusion_reasons": r["exclusion_reasons"]} for r in rows),
              ["case_id", "partition", "cohort", "exclusion_reasons"])
    print(json.dumps({"status": result["status"], "counts": result["counts"], "output": str(args.output)}, ensure_ascii=False))


def numeric_summary(values, positive_only=False):
    observed = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number) and (not positive_only or number > 0):
            observed.append(number)
    return {"observed": len(observed), "missing_or_invalid": len(values) - len(observed),
            "minimum": min(observed) if observed else None, "median": float(np.median(observed)) if observed else None,
            "maximum": max(observed) if observed else None}


def assess_cohorts(cases, metadata, reviews, labels):
    """Development 후보 상한만 계산한다. pending·uncertain을 승인으로 바꾸지 않는다."""
    if any(row["official_split"] != "train" for row in cases.values()):
        raise ValueError("Cohort assessment must contain development cases only")
    if not cases.keys() <= metadata.keys():
        raise ValueError("Missing metadata for a development case")
    profiles = {}
    predicates = {
        "any_panel_nonempty": lambda row: int(row["nonempty_panel_organs"]) >= 1,
        "all_panel_nonempty": lambda row: int(row["nonempty_panel_organs"]) == len(labels),
        "all_panel_nonempty_no_border": lambda row: int(row["nonempty_no_border_organs"]) == len(labels),
    }
    for name, predicate in predicates.items():
        groups = {cohort: [case for case, row in cases.items() if row["cohort"] == cohort and predicate(row)]
                  for cohort in ("adult", "pediatric")}
        profiles[name] = {}
        for cohort, ids in groups.items():
            meta = [metadata[case] for case in ids]
            profiles[name][cohort] = {
                "automatic_candidate_count": len(ids),
                "case_ids_with_rejected_annotation": sorted(case for case in ids if any(r["annotation"] == "reject" for r in reviews[case])),
                "case_ids_with_partial_or_out_of_fov_review": sorted(case for case in ids if any(r["coverage"] in {"partial", "out_of_fov"} for r in reviews[case])),
                "recorded_full_and_acceptable_all_organs": sum(len(reviews[case]) == len(labels)
                    and all(r["coverage"] == "full" and r["annotation"] == "acceptable" for r in reviews[case]) for case in ids),
                "age": numeric_summary([r["age"] for r in meta]),
                "source_slice_thickness": numeric_summary([r["slice_thickness"] for r in meta], positive_only=True),
                "categorical_metadata": {field: dict(collections.Counter(r[field].strip() or "<missing>" for r in meta))
                                         for field in ("institute", "study_type", "gender")},
            }
    budgets = []
    for profile, cohorts in profiles.items():
        sizes = {cohort: row["automatic_candidate_count"] - len(row["case_ids_with_rejected_annotation"])
                 for cohort, row in cohorts.items()}
        for validation in (16, 32):
            for k in (16, 32, 64, 128):
                counts = allocation_counts(sizes, k, validation)
                budgets.append({"profile": profile, "K": k, "validation_per_cohort": validation,
                                "arithmetic_feasible_before_remaining_QC": counts is not None,
                                "counts_if_all_remaining_candidates_qualify": counts})
    return {"status": "development_assessment_only_not_a_split", "profiles": profiles, "budget_scenarios": budgets,
            "limitations": ["Automatic screens are upper bounds, not accepted cohorts or power calculations.",
                            "Budget calculations remove recorded annotation rejections but do not assume other candidates will pass full-FOV review.",
                            "Missing institution/study_type cannot be used to claim matched acquisition cohorts.",
                            "Source slice_thickness is not the resampled NIfTI voxel spacing.",
                            "Patient independence cannot be established from case IDs and exact CT hashes alone.",
                            "No test records, model predictions, or performance scores are used in this assessment."]}


def assess(args):
    from .review import load_review_ledger, panel_labels
    study = json.loads(args.study.read_text())
    summary, cases, reviews, _ = load_review_ledger(study, args.review, args.qc)
    _, metadata, _, _, provenance = load_inputs(args.raw, DEFAULT_INVENTORY, ROOT / study["qc_protocol"])
    for key, value in provenance.items():
        if summary["provenance"].get(key) != value:
            raise ValueError("Assessment source provenance mismatch: " + key)
    result = assess_cohorts(cases, metadata, reviews, panel_labels(study))
    result["planned_allocation"] = {key: study["cohort"].get(key) for key in
                                    ("training_candidate_screen", "finetuning_cases_per_arm", "validation_cases_per_cohort",
                                     "allocation_status", "split_unit", "patient_independence_verified")}
    plan = result["planned_allocation"]
    profile = result["profiles"].get(plan["training_candidate_screen"])
    if profile is not None and plan["finetuning_cases_per_arm"] is not None and plan["validation_cases_per_cohort"] is not None:
        plan["upper_bound_counts_if_all_candidates_pass_remaining_QC"] = allocation_counts(
            {cohort: row["automatic_candidate_count"] - len(row["case_ids_with_rejected_annotation"]) for cohort, row in profile.items()},
            plan["finetuning_cases_per_arm"], plan["validation_cases_per_cohort"])
    result["provenance"] = {**provenance, "study_sha256": sha256(args.study), "review_csv_sha256": sha256(args.review / "review.csv"),
                            "cases_csv_sha256": sha256(args.review / "cases.csv"), "assessment_code_sha256": sha256(__file__),
                            "review_code_sha256": sha256(Path(__file__).with_name("review.py")), **git_provenance()}
    result["created_at"] = datetime.datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
    # A regenerated assessment is a report, not an immutable allocation or a manually edited ledger.
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "cohort_assessment.json", result)
    lines = ["# Development cohort 및 K 검토", "", result["created_at"], "",
             "이 보고서는 자동 screen의 표본 상한을 비교한다. 최종 포함 판정이나 split을 생성하지 않았다.", "",
             "| 자동 screen | 성인 후보 | 소아 후보 |", "|---|---:|---:|"]
    for name, groups in result["profiles"].items():
        lines.append(f"| {name} | {groups['adult']['automatic_candidate_count']} | {groups['pediatric']['automatic_candidate_count']} |")
    lines += ["", "## 현재 계획값", "",
              f"분할 단위: {plan['split_unit']}. 환자 독립성 확인: {plan['patient_independence_verified']}.",
              f"M1·M2 각 {plan['finetuning_cases_per_arm']}건, validation 각 {plan['validation_cases_per_cohort']}건을 목표로 한다.",
              f"우선 검토 screen: {plan['training_candidate_screen']}. 상태: {plan['allocation_status']}.",
              "이 수는 연구 계획값이며 실제 case 배정이 아니다. 최종 QC에서 부족하면 protocol을 명시적으로 수정해야 한다."]
    lines += ["", "## K 시나리오", "", "아래 수는 동일한 validation 크기를 두 cohort에 적용하고 기록된 annotation reject를 제외한 산술 결과다.",
              "남은 수동 검토·환자 grouping 이후 줄어들 수 있으며, 최소 M0 1건을 남긴다는 계산만 하므로 통계적 충분성을 뜻하지 않는다.", "",
              "| screen | validation/군 | 가능한 K 후보 |", "|---|---:|---|"]
    for profile in result["profiles"]:
        for validation in (16, 32):
            available = [str(r["K"]) for r in result["budget_scenarios"] if r["profile"] == profile
                         and r["validation_per_cohort"] == validation and r["arithmetic_feasible_before_remaining_QC"]]
            lines.append(f"| {profile} | {validation} | {', '.join(available) or '없음'} |")
    lines += ["", "## Metadata 결측과 차이", "", "| screen | cohort | 기관 결측 | 촬영 종류 결측 | 원본 slice thickness 중앙값 |", "|---|---|---:|---:|---:|"]
    for name, groups in result["profiles"].items():
        for cohort, row in groups.items():
            categories = row["categorical_metadata"]
            lines.append(f"| {name} | {cohort} | {categories['institute'].get('<missing>', 0)}/{row['automatic_candidate_count']} | "
                         f"{categories['study_type'].get('<missing>', 0)}/{row['automatic_candidate_count']} | {row['source_slice_thickness']['median']} |")
    recorded = result["profiles"]["any_panel_nonempty"]
    recorded_counts = {cohort: row["recorded_full_and_acceptable_all_organs"] for cohort, row in recorded.items()}
    lines += ["", "## 아직 확정할 수 없는 항목", "",
              "- 환자 식별자: 현 metadata에 없어 patient-level 독립성을 보장할 수 없다. 사용자 확인에 따라 case 단위 탐색 연구로 진행하며, case bootstrap의 독립성 가정과 잔여 환자 중복 가능성을 한계로 기록한다.",
              f"- 판정: 모든 선정 장기가 full·acceptable로 기록된 후보는 성인 {recorded_counts['adult']}건, 소아 {recorded_counts['pediatric']}건이다. pending·uncertain을 승인으로 바꾸지 않았다.",
              "- 촬영 조건: 특히 소아의 기관·촬영 종류 결측이 커서 해당 변수의 matching을 수행했다고 주장할 수 없다.",
              "- K·validation: 시나리오 표는 대안 비교다. study.json의 현재 목표값도 실제 선정 case의 QC 완료와 split 고정을 의미하지 않는다.",
              "- 단위·최종 전처리·학습 budget·NSD 허용 거리: 여전히 별도 근거가 필요하다.",
              "", "기계 판독 결과와 입력 hash: `cohort_assessment.json`. 성능·유의성을 보고 유리한 집합을 고른 것이 아니다."]
    (args.output / "cohort_assessment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "report": str(args.output / "cohort_assessment.md")}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qc", type=Path, default=ROOT / "manifests/qc_v300")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=ROOT / "manifests/splits_v300")
    parser.add_argument("--provisional", action="store_true", help="Explicitly permit case-level candidate splits; no patient-level claim")
    parser.add_argument("--assess", action="store_true", help="Only assess development cohort sizes/covariates; do not allocate cases")
    parser.add_argument("--freeze-development", action="store_true", help="Freeze the adopted public-reference development split; test organ eligibility remains pending")
    parser.add_argument("--study", type=Path, default=ROOT / "configs/study.json")
    parser.add_argument("--review", type=Path, default=ROOT / "manifests/review_v300")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    args = parser.parse_args()
    if sum((args.assess, args.provisional, args.freeze_development)) > 1:
        parser.error("Choose exactly one split operation")
    if args.freeze_development:
        if args.output == ROOT / "manifests/splits_v300":
            args.output = ROOT / "manifests" / ("splits_" + json.loads(args.study.read_text())["study_version"])
        freeze_development(args)
        return
    if args.assess:
        if args.provisional:
            parser.error("--assess and --provisional are mutually exclusive")
        if args.output == ROOT / "manifests/splits_v300":
            args.output = ROOT / "outputs/review_v300"
        assess(args)
        return
    settings = json.loads(args.protocol.read_text(encoding="utf-8"))
    summary = json.loads((args.qc / "summary.json").read_text(encoding="utf-8"))
    if summary["completed_cases"] != summary["expected_cases"] or summary["status"] == "partial":
        raise ValueError("Full automated QC must complete before splitting")
    if summary["provenance"]["protocol_sha256"] != sha256(args.protocol):
        raise ValueError("QC and split protocols differ")
    if not args.provisional:
        raise ValueError("This legacy splitter only supports --provisional. The current study requires organ-reviewed eligibility before a case-level split can be frozen")
    rows = read_csv(args.qc / "case_qc.csv")
    pools = make_splits(rows, settings)
    if args.output.exists():
        raise FileExistsError("Split outputs are immutable; choose a new version directory")
    args.output.mkdir(parents=True)
    provenance = {"qc_summary_sha256": sha256(args.qc / "summary.json"),
                  "case_qc_sha256": sha256(args.qc / "case_qc.csv"),
                  "protocol_sha256": sha256(args.protocol), "split_code_sha256": sha256(__file__),
                  **git_provenance()}
    result = {"status": settings["split_status"], "seed": settings["seed"],
              "unit": "case_with_exact_ct_deduplication_not_verified_patient", "provenance": provenance,
              "counts": {k: len(v) for k, v in pools.items()}, "partitions": pools}
    write_json(args.output / "splits.json", result)
    assignments = {case_id: group for group, cases in pools.items() for case_id in cases}
    write_csv(args.output / "assignments.csv", (
        {"case_id": r["case_id"], "partition": assignments[r["case_id"]], "cohort": r["cohort"],
         "exclusion_reasons": r["exclusion_reasons"]} for r in rows),
        ["case_id", "partition", "cohort", "exclusion_reasons"])
    print(json.dumps({"status": result["status"], "counts": result["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
