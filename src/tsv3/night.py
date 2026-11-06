"""전처리 완료를 기다린 뒤 cache 검증과 짧은 GPU 통합 시험을 수행한다. 본 학습은 실행하지 않는다."""

import argparse
import fcntl
import json
import os
import random
import resource
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

from .data import ROOT, DEFAULT_RAW, sha256, git_provenance
from .preprocessing import load_plan, run_cache, atomic_json, normalize_inplace, check_space


def cache_writer_active(cache):
    with (cache / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def wait_for_writer(cache, timeout_seconds, report):
    deadline = time.monotonic() + timeout_seconds
    while cache_writer_active(cache):
        if time.monotonic() >= deadline:
            raise TimeoutError("Preprocessing is still active; overnight wait limit reached")
        manifest = json.loads((cache / "manifest.json").read_text())
        report("waiting_for_preprocessing", completed=len(manifest["completed"]), expected=manifest["expected_cases"])
        time.sleep(30)
    manifest = json.loads((cache / "manifest.json").read_text())
    allowed = {"running", "interrupted", "partial_limit_not_training_ready", "development_cache_complete_not_training_results"}
    if manifest["status"] not in allowed:
        raise ValueError("Preprocessing failed; no automatic retry of data errors: " + str(manifest.get("error")))
    return manifest


def verify_completion(cache, plan):
    manifest = json.loads((cache / "manifest.json").read_text())
    if manifest["status"] != "development_cache_complete_not_training_results" or manifest["error"] is not None:
        raise ValueError("Development cache is incomplete")
    if manifest["signature"] != plan["signature"] or set(manifest["completed"]) != set(plan["roles"]):
        raise ValueError("Cache contract/case set differs from frozen split")
    for case, record in manifest["completed"].items():
        props = record["properties"]
        if record["role"] != plan["roles"][case] or props["training_target_nonempty_labels"] != len(plan["labels"]) or props["labels_lost_after_overlap_handling_and_resampling"]:
            raise ValueError("Case role or label preservation mismatch: " + case)
    norm_record = manifest["normalization"]
    if norm_record["file"] != "normalization.json" or sha256(cache / "normalization.json") != norm_record["sha256"]:
        raise ValueError("Normalization file hash mismatch")
    normalization = json.loads((cache / "normalization.json").read_text())
    if normalization["fit_cases"] != sorted(plan["split"]["partitions"]["M0_train"]):
        raise ValueError("Normalization must use exactly M0 train")
    if normalization["split_sha256"] != plan["signature"]["split_sha256"]:
        raise ValueError("Normalization split hash mismatch")
    if not all(np.isfinite(normalization[k]) for k in ("lower", "upper", "mean", "std")) or normalization["std"] <= 0 or normalization["lower"] >= normalization["upper"]:
        raise ValueError("Invalid normalization statistics")
    return manifest, normalization


def integration_test(cache, plan, output, report):
    import torch
    from .training import (build_model, state_hash, optimization_components, sample_patch, update_step,
                           save_checkpoint, load_checkpoint, train_interval, infer_volume)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for scheduled integration test")
    manifest, normalization = verify_completion(cache, plan)
    study = plan["study"]
    config = json.loads((ROOT / "configs/probe.json").read_text())
    config.update(study["training"]["feasibility_probe_overrides"])
    config["seed"] = study["training"]["paired_training_seeds"][0]
    config["purpose"] = "cached_development_integration_test_not_research_training"
    for key in ("patch_size", "batch_size", "features_per_stage"):
        config[key] = study["training"]["fixed_execution_plan"][key]
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Torch 2.8 CUDA NLL reduction is not deterministic. Require exact restored
    # inputs and a bounded numerical comparison, never claim unconditional bitwise replay.
    torch.use_deterministic_algorithms(True, warn_only=True)

    def representative(role):
        ids = sorted(plan["split"]["partitions"][role], key=lambda c: (np.prod(manifest["completed"][c]["properties"]["preprocessed_shape"]), c))
        return ids[len(ids) // 2]

    selected = {role: representative(role) for role in
                ("M0_train", "M1_finetune", "M2_finetune", "adult_validation", "pediatric_validation")}
    if len(set(selected.values())) != 5 or not set(selected.values()) <= set(plan["roles"]):
        raise ValueError("Invalid or overlapping development integration cases")
    provenance = {"split_sha256": plan["signature"]["split_sha256"],
                  "normalization_sha256": manifest["normalization"]["sha256"]}
    result = {"status": "running", "purpose": config["purpose"], "config": config,
              "selected_cases": selected, "provenance": provenance, "training": {}, "inference": {},
              "loading_seconds": {}, "torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(), "research_training_started": False,
              "determinism": "cudnn_deterministic_and_torch_warn_only_CUDA_NLL_reduction_not_guaranteed_bitwise"}
    start = time.monotonic()

    def read_case(case):
        begin = time.monotonic()
        with np.load(cache / manifest["completed"][case]["file"], allow_pickle=False) as arrays:
            data, seg = arrays["data"], arrays["seg"]
        if data.dtype != np.float32 or seg.dtype != np.uint8 or data.shape != seg.shape or data.shape[0] != 1:
            raise ValueError("Invalid cache arrays: " + case)
        normalize_inplace(data, normalization)
        if not np.isfinite(data).all():
            raise ValueError("Nonfinite normalized CT")
        result["loading_seconds"][case] = time.monotonic() - begin
        return data, seg

    model = build_model(config, len(plan["labels"]) + 1)
    ignore = len(plan["labels"]) + 1
    try:
        # The same patch/step implementation is used by the legacy probe and checkpoint test.
        case = selected["M0_train"]
        data, seg = read_case(case)
        optimizer, loss_fn, scaler = optimization_components(model, config, ignore)
        rng = np.random.default_rng(config["seed"])
        initial = state_hash(model.state_dict())
        step_times = []
        losses = []
        torch.cuda.reset_peak_memory_stats()
        updates = config["warmup_steps"] + config["measured_steps"]
        for step in range(updates):
            torch.cuda.synchronize()
            tick = time.monotonic()
            x, y = sample_patch(data, seg, config, rng, ignore)
            loss, skipped = update_step(model, x, y, optimizer, loss_fn, scaler)
            if skipped:
                raise RuntimeError("AMP skipped an integration optimizer update")
            torch.cuda.synchronize()
            if step >= config["warmup_steps"]:
                step_times.append(time.monotonic() - tick)
                losses.append(loss)
            del x, y
        m0_hash = state_hash(model.state_dict())
        if m0_hash == initial:
            raise ValueError("Model weights did not change")
        result["training"]["M0_probe"] = {"optimizer_updates": updates, "finite_loss": True,
                  "median_step_seconds": float(np.median(step_times)), "losses_not_accuracy": losses,
                  "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                  "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20, "state_sha256": m0_hash}
        with tempfile.TemporaryDirectory(prefix=".checkpoint-test-", dir=output) as tmp:
            checkpoint = Path(tmp) / "resume.pt"
            save_checkpoint(checkpoint, model, optimizer, scaler, rng, updates, config, provenance)
            checkpoint_hash = sha256(checkpoint)
            x, y = sample_patch(data, seg, config, rng, ignore)
            reference_patch_hash = state_hash({"data": x, "target": y})
            reference_loss, skipped = update_step(model, x, y, optimizer, loss_fn, scaler)
            if skipped:
                raise RuntimeError("AMP skipped uninterrupted reference update")
            reference_hash = state_hash(model.state_dict())
            reference_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            del x, y, optimizer, scaler, loss_fn, model
            torch.cuda.empty_cache()
            model = build_model(config, len(plan["labels"]) + 1)
            optimizer, loss_fn, scaler = optimization_components(model, config, ignore)
            rng = np.random.default_rng(0)
            restored_updates = load_checkpoint(checkpoint, model, optimizer, scaler, rng, config, provenance)
            if restored_updates != updates or state_hash(model.state_dict()) != m0_hash:
                raise ValueError("Checkpoint failed to restore model/update count")
            x, y = sample_patch(data, seg, config, rng, ignore)
            if state_hash({"data": x, "target": y}) != reference_patch_hash:
                raise ValueError("Checkpoint did not restore the next sampled patch exactly")
            resumed_loss, skipped = update_step(model, x, y, optimizer, loss_fn, scaler)
            maximum_difference = max(float((value.detach().cpu() - reference_state[key]).abs().max())
                                     for key, value in model.state_dict().items())
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value.detach().cpu(), reference_state[key], rtol=1e-5, atol=1e-6)
            if skipped or not np.isclose(resumed_loss, reference_loss, rtol=1e-5, atol=1e-6):
                raise ValueError("Resumed next update differs from uninterrupted reference")
            result["checkpoint_resume"] = {"identical_next_patch": True, "next_update_matches_within_tolerance": True,
                                           "bitwise_identical_weights": state_hash(model.state_dict()) == reference_hash,
                                           "rtol": 1e-5, "atol": 1e-6, "max_parameter_abs_difference": maximum_difference,
                                           "loss_abs_difference": abs(resumed_loss - reference_loss),
                                           "restored_updates": restored_updates, "checkpoint_sha256": checkpoint_hash}
            load_checkpoint(checkpoint, model, optimizer, scaler, rng, config, provenance)
            m0_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            del x, y, data, seg, optimizer, scaler, loss_fn, reference_state
        report("integration_resume_passed")

        for branch, role in (("M0_probe", "M0_train"), ("M1_probe", "M1_finetune"), ("M2_probe", "M2_finetune")):
            model.load_state_dict(m0_state)
            if state_hash(model.state_dict()) != m0_hash:
                raise ValueError("Branch does not start from identical M0 weights")
            if branch != "M0_probe":
                case = selected[role]
                data, seg = read_case(case)
                measured = train_interval(model, [case], {case: (data, seg)}, config, ignore)
                measured["initial_state_sha256"] = m0_hash
                measured["weights_changed"] = state_hash(model.state_dict()) != m0_hash
                if measured["optimizer_updates"] != updates or measured["skipped_amp_steps"] or not measured["weights_changed"]:
                    raise ValueError("Unequal or ineffective fine-tuning test updates")
                result["training"][branch] = measured
                del data, seg
            result["inference"][branch] = []
            for validation in ("adult_validation", "pediatric_validation"):
                case = selected[validation]
                data, seg = read_case(case)
                prediction, timing = infer_volume(model, data, plan["labels"], config)
                if prediction.shape != tuple(data.shape[1:]) or prediction.min() < 0 or prediction.max() > len(plan["labels"]):
                    raise ValueError("Invalid full-volume prediction")
                timing["case_id"] = case
                result["inference"][branch].append(timing)
                del data, seg, prediction
            atomic_json(output / "result.json", result)
            report("integration_branch_complete", branch=branch)
        result["status"] = "integration_passed_research_training_not_started"
    except BaseException as exc:
        result.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - start
        result["peak_process_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        result["limitations"] = ["Short development integration test; not trained research M0/M1/M2.",
                                 "Five representative development cases; no test data or accuracy selection.",
                                 "Checkpoint numerical agreement is tested on this environment; CUDA reduction can be nondeterministic.",
                                 "Training budget, augmentation and final evaluation contract remain pending."]
        atomic_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "configs/study.json")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--review", type=Path, default=ROOT / "manifests/review_v300")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/preprocessed/abdominal_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/overnight_abdominal_v1")
    parser.add_argument("--host-volume", type=Path, default=None)
    parser.add_argument("--release-sleep-flag", type=Path)
    parser.add_argument("--wait-hours", type=float, default=12)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0 < args.wait_hours <= 24:
        parser.error("wait-hours must be in (0,24]")
    plan = load_plan(args)
    if args.dry_run:
        print(json.dumps({"status": "plan_only", "cases": len(plan["roles"]),
                          "sequence": ["wait_or_resume_preprocessing", "verify_all_cache_and_inputs", "GPU_checkpoint_and_inference_integration"],
                          "research_training": False, "output": str(args.output)}))
        return
    if (args.output / "status.json").exists():
        raise FileExistsError("Overnight run exists; use a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    initial_hashes = {str(path): sha256(path) for path in
                     (args.study, ROOT / "uv.lock", Path(__file__), Path(__file__).with_name("training.py"),
                      Path(__file__).with_name("preprocessing.py"), Path(__file__).with_name("audit.py"))}
    state = {"status": "starting", "pid": os.getpid(), "research_training_started": False,
             "code_and_config_hashes": initial_hashes, "provenance": git_provenance()}
    def report(stage, **details):
        state.update(status=stage, updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **details)
        atomic_json(args.output / "status.json", state)
        print(json.dumps({"status": stage, **details}, ensure_ascii=False), flush=True)
    def unchanged():
        if any(sha256(path) != expected for path, expected in initial_hashes.items()):
            raise ValueError("Overnight code/config changed after scheduling")
    try:
        report("starting")
        wait_for_writer(args.cache, args.wait_hours * 3600, report)
        unchanged()
        report("verifying_or_resuming_cache")
        cache_args = SimpleNamespace(raw=args.raw, output=args.cache, resume=True, limit=0,
                                     host_volume=args.host_volume, reserve_gib=10)
        run_cache(cache_args, plan)
        manifest, _ = verify_completion(args.cache, plan)
        state["cache_verification"] = {"cases": len(manifest["completed"]),
                                      "normalization_fit_cases": manifest["normalization"]["fit_case_count"],
                                      "manifest_sha256": sha256(args.cache / "manifest.json")}
        unchanged()
        check_space(args.output, args.host_volume, 1 * 2**30, 10)
        report("running_GPU_integration")
        integration_test(args.cache, plan, args.output, report)
        report("complete_preprocessing_and_integration_only", completed=len(plan["roles"]))
    except BaseException as exc:
        report("failed", error=type(exc).__name__ + ": " + str(exc))
        raise
    finally:
        if args.release_sleep_flag is not None:
            args.release_sleep_flag.touch()


if __name__ == "__main__":
    main()
