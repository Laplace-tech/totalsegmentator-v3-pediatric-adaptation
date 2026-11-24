"""고정 development validation을 선언된 seed 순서로 집계한다.

Usage:
    .venv/bin/python scripts/02_summarize_development.py --seeds 20261029 20261030 20261031

공식 test 입력은 읽지 않는다. 기존 branch validation JSON과 M0 소아 사후 평가만 사용한다.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from tsv3.data import ROOT
from tsv3.preprocessing import atomic_json
from tsv3.training import branch_config, state_hash


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validated_scores(block, expected_cases, organs):
    if sorted(block["cases"]) != expected_cases:
        raise ValueError("Validation case IDs differ from frozen split")
    values = np.asarray([block["cases"][case] for case in expected_cases], dtype=np.float64)
    if values.shape != (len(expected_cases), len(organs)) or not np.isfinite(values).all():
        raise ValueError("Invalid validation Dice shape or nonfinite score")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("Validation Dice outside [0, 1]")
    if not math.isclose(float(values.mean()), block["macro_Dice"], abs_tol=1e-12):
        raise ValueError("Stored macro Dice differs from case scores")
    if not np.allclose(values.mean(axis=0), block["organ_mean_Dice"], rtol=0, atol=1e-12):
        raise ValueError("Stored organ Dice differs from case scores")
    return values


def summarize(seeds, study_path=ROOT / "configs/study.json"):
    study = read_json(study_path)
    planned = study["training"]["replication_plan"]["seeds"]
    if len(seeds) < 2 or seeds != planned[:len(seeds)]:
        raise ValueError("Report all completed seeds in declared order; no seed selection")
    split = read_json(ROOT / study["cohort"]["split_manifest"])
    organs = [label for label, index in sorted(study["labels"].items(), key=lambda item: item[1]) if index]
    case_ids = {cohort: sorted(split["partitions"][cohort + "_validation"])
                for cohort in ("adult", "pediatric")}
    scores, arrays, integrity, inputs, identities = {}, {}, {}, {}, []
    numeric_reference = {}

    for seed in seeds:
        directory = ROOT / f"outputs/training_{study['study_version']}_seed{seed}"
        run_path = directory / "run.json"
        run = read_json(run_path)
        provenance = run["provenance"]
        run_seed = provenance.get("training_seed", run["study_snapshot"]["training"]["paired_training_seeds"][0])
        if run["status"] != "complete" or run_seed != seed:
            raise ValueError(f"Incomplete or wrong-seed run: {seed}")
        snapshot = directory / "source_snapshot.tar.gz"
        if digest(snapshot) != run["source_snapshot_sha256"]:
            raise ValueError(f"Source snapshot changed: {seed}")
        identities.append((provenance["split_sha256"], provenance["normalization_sha256"],
                           provenance["evaluation_identity"]["contract_sha256"],
                           provenance["evaluation_identity"]["eligibility_sha256"]))
        scores[str(seed)], arrays[seed] = {}, {}
        integrity[str(seed)] = {"run_complete": True, "budget_updates": {}, "selected_sha256": {},
                                "initial_state_sha256": {}, "source_snapshot_sha256": run["source_snapshot_sha256"],
                                "study_sha256": provenance["study_sha256"]}
        inputs[str(run_path.relative_to(ROOT))] = digest(run_path)

        for model in ("M0", "M1", "M2"):
            status_path = directory / model / "status.json"
            status = read_json(status_path)
            config = branch_config(run["study_snapshot"], model, False, seed)
            budget = config["optimizer_updates"]
            if (status["status"] != "complete" or status["updates"] != budget or status["budget"] != budget
                    or status.get("error")):
                raise ValueError(f"Incomplete branch: {seed}/{model}")
            common_config = {key: value for key, value in config.items() if key != "seed"}
            if model in numeric_reference and numeric_reference[model] != common_config:
                raise ValueError(f"Scientific config differs across seeds: {model}")
            numeric_reference.setdefault(model, common_config)
            selected = directory / model / "selected.pt"
            if digest(selected) != status["selected_checkpoint_sha256"]:
                raise ValueError(f"Selected checkpoint changed: {seed}/{model}")
            integrity[str(seed)]["budget_updates"][model] = budget
            integrity[str(seed)]["selected_sha256"][model] = status["selected_checkpoint_sha256"]
            integrity[str(seed)]["initial_state_sha256"][model] = status["trainer_state"]["initial_state_sha256"]
            inputs[str(status_path.relative_to(ROOT))] = digest(status_path)
            scores[str(seed)][model], arrays[seed][model] = {}, {}

            selected_update = status["trainer_state"]["best_update"] if model == "M0" else budget
            if type(selected_update) is not int or not 0 < selected_update <= budget:
                raise ValueError(f"Invalid selected update: {seed}/{model}")
            validation_path = directory / model / f"validation_{selected_update:06d}.json"
            validation = read_json(validation_path)
            inputs[str(validation_path.relative_to(ROOT))] = digest(validation_path)
            if model == "M0":
                pediatric_path = directory / model / "pediatric_validation_selected.json"
                pediatric = read_json(pediatric_path)
                if (pediatric["status"] != "posthoc_development_validation_no_model_selection"
                        or pediatric["checkpoint_sha256"] != status["selected_checkpoint_sha256"]):
                    raise ValueError(f"M0 pediatric evaluation used wrong checkpoint: {seed}")
                validation["pediatric_validation"] = pediatric["pediatric_validation"]
                inputs[str(pediatric_path.relative_to(ROOT))] = digest(pediatric_path)

            for cohort in ("adult", "pediatric"):
                values = validated_scores(validation[cohort + "_validation"], case_ids[cohort], organs)
                arrays[seed][model][cohort] = values
                scores[str(seed)][model][cohort] = {
                    "macro_Dice": float(values.mean()),
                    "organ_mean_Dice": dict(zip(organs, map(float, values.mean(axis=0))))}
        starts = integrity[str(seed)]["initial_state_sha256"]
        if starts["M1"] != starts["M2"]:
            raise ValueError(f"M1/M2 initial states differ: {seed}")
        m0_checkpoint = directory / "M0/selected.pt"
        m0_payload = torch.load(m0_checkpoint, map_location="cpu", weights_only=False)
        m0_state_sha = state_hash(m0_payload["model"])
        m0_status = read_json(directory / "M0/status.json")
        if (m0_payload["updates"] != m0_status["trainer_state"]["best_update"]
                or m0_payload["config"] != branch_config(run["study_snapshot"], "M0", False, seed)):
            raise ValueError(f"M0 selected checkpoint content differs: {seed}")
        del m0_payload
        integrity[str(seed)]["M0_state_sha256"] = m0_state_sha
        for model in ("M1", "M2"):
            payload = torch.load(directory / model / "selected.pt", map_location="cpu", weights_only=False)
            branch_provenance = payload["provenance"]
            if (payload["updates"] != integrity[str(seed)]["budget_updates"][model]
                    or payload["config"] != branch_config(run["study_snapshot"], model, False, seed)
                    or branch_provenance["M0_checkpoint_sha256"] != integrity[str(seed)]["selected_sha256"]["M0"]
                    or branch_provenance["M0_state_sha256"] != m0_state_sha
                    or payload["trainer_state"]["initial_state_sha256"] != m0_state_sha):
                raise ValueError(f"Branch does not start from selected M0: {seed}/{model}")
            del payload
        integrity[str(seed)]["M1_M2_same_M0_selected_initialization_and_numeric_config"] = True

    if len(set(identities)) != 1:
        raise ValueError("Split, normalization or evaluation contract differs across seeds")
    comparisons = {}
    for cohort in ("adult", "pediatric"):
        for left, right in (("M1", "M2"), ("M1", "M0"), ("M2", "M0")):
            per_seed, seed_means, organ_deltas, case_deltas = {}, [], [], []
            for seed in seeds:
                delta = arrays[seed][left][cohort] - arrays[seed][right][cohort]
                by_case, by_organ = delta.mean(axis=1), delta.mean(axis=0)
                seed_means.append(float(delta.mean()))
                organ_deltas.append(by_organ)
                case_deltas.append(by_case)
                per_seed[str(seed)] = {"macro_delta": float(delta.mean()),
                    "organ_delta": dict(zip(organs, map(float, by_organ))),
                    "case_macro_delta": dict(zip(case_ids[cohort], map(float, by_case))),
                    "cases_positive": int((by_case > 0).sum()),
                    "cases_negative": int((by_case < 0).sum())}
            organ_deltas = np.asarray(organ_deltas)
            averaged_case = np.mean(case_deltas, axis=0)
            rng = np.random.default_rng(study["evaluation"]["uncertainty"]["seed"])
            repeats = study["evaluation"]["uncertainty"]["bootstrap_replicates"]
            resampled = averaged_case[rng.integers(0, len(averaged_case),
                                                  size=(repeats, len(averaged_case)))].mean(axis=1)
            comparisons[f"{cohort}_{left}_minus_{right}"] = {
                "per_seed": per_seed,
                "seed_macro_mean": float(np.mean(seed_means)),
                "seed_macro_sample_SD": float(np.std(seed_means, ddof=1)),
                "organ_delta_mean": dict(zip(organs, map(float, organ_deltas.mean(axis=0)))),
                "organ_delta_sample_SD": dict(zip(organs, map(float, organ_deltas.std(axis=0, ddof=1)))),
                "organ_positive_seed_counts": dict(zip(organs, map(int, (organ_deltas > 0).sum(axis=0)))),
                "organ_negative_seed_counts": dict(zip(organs, map(int, (organ_deltas < 0).sum(axis=0)))),
                "paired_case_bootstrap_seed_average": {
                    "replicates": repeats, "case_unit": len(averaged_case),
                    "seed_values_kept_together": True, "point": float(averaged_case.mean()),
                    "pointwise_percentile_95": list(map(float, np.quantile(resampled, [0.025, 0.975]))),
                    "exploratory_validation_only": True}}
    return {"status": f"{len(seeds)}_of_{len(planned)}_seed_development_validation_descriptive",
            "seeds": seeds, "planned_seeds": planned, "models": ["M0", "M1", "M2"],
            "organs": organs, "case_ids": case_ids, "input_file_sha256": inputs,
            "common_input_identity": dict(zip(("split_sha256", "normalization_sha256",
                                               "evaluation_contract_sha256", "eligibility_sha256"), identities[0])),
            "model_scores": scores, "comparisons": comparisons, "integrity": integrity,
            "limitations": ["same development cases reused across seeds and M0 checkpoint selection",
                            "seed outputs are not independent patients",
                            "patient linkage cannot be verified from supplied data",
                            "no official test predictions or scores used"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(args.seeds)
    output = args.output or ROOT / f"outputs/training_abdominal_v1_seed{args.seeds[-1]}/replication_validation.json"
    if output.exists():
        if read_json(output) != report:
            raise FileExistsError(f"Existing report differs; preserve and investigate: {output}")
        status = "existing_identical_report"
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(output, report)
        status = "report_written"
    print(json.dumps({"status": status, "output": str(output), "sha256": digest(output),
                      "seeds": args.seeds,
                      "pediatric_M1_minus_M2": report["comparisons"]["pediatric_M1_minus_M2"]["seed_macro_mean"],
                      "pediatric_M1_minus_M0": report["comparisons"]["pediatric_M1_minus_M0"]["seed_macro_mean"],
                      "adult_M1_minus_M0": report["comparisons"]["adult_M1_minus_M0"]["seed_macro_mean"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
