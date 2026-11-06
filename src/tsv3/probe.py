"""실제 development CT로 동일 3D U-Net의 세 분기 실행 비용만 측정한다."""

import argparse
import importlib.metadata
import json
import os
import platform
import random
import resource
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("nnUNet_compile", "false")

import nibabel as nib
import numpy as np
import torch
from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape

from .data import (ROOT, DEFAULT_RAW, DEFAULT_INVENTORY, DEFAULT_PROTOCOL, load_inputs,
                   read_csv, sha256, write_json, git_provenance)
from .splits import validate_splits
from .preprocessing import preprocess_case, fit_normalization, normalize_inplace
from .training import architecture, build_model, state_hash, train_interval, infer_volume


def throughput_probe(args):
    """Replay a paused M0 checkpoint; never write research weights or touch test data."""
    import contextlib
    import fcntl
    import hashlib
    from . import training as train
    from .preprocessing import load_plan, atomic_json
    from .night import verify_completion
    from .fast_cache import verify_fast_cache
    import blosc2

    if args.run is None or not 20 <= args.steps <= 1000:
        raise ValueError("--throughput requires --run and 20 <= --steps <= 1000")
    args.study = args.study or ROOT / "configs/study.json"
    if args.output == ROOT / "outputs/feasibility_v300":
        args.output = ROOT / "outputs" / args.run.name.replace("training_", "throughput_", 1)
    if args.output.resolve() == args.run.resolve() or args.output.resolve().is_relative_to(args.run.resolve()):
        raise ValueError("Benchmark output must be separate from the research run")
    with contextlib.ExitStack() as locks:
        for root, mode in ((args.run, fcntl.LOCK_EX), (args.cache, fcntl.LOCK_SH), (args.fast_cache, fcntl.LOCK_SH)):
            lock = locks.enter_context((root / ".lock").open("a"))
            try:
                fcntl.flock(lock, mode | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Stop research training/cache maintenance before benchmarking") from exc
        previous = json.loads((args.run / "run.json").read_text())
        if previous["status"] != "interrupted" or (args.run / "M1").exists() or (args.run / "M2").exists():
            raise ValueError("This benchmark requires a paused M0-only research run")
        plan = load_plan(args)
        manifest, normalization = verify_completion(args.cache, plan)
        fast = verify_fast_cache(args.fast_cache, args.cache, manifest)
        checkpoint = args.run / "M0/latest.pt"
        digest = sha256(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = payload["config"]
        old = previous["provenance"]
        sources = {p.name: sha256(p) for p in sorted(Path(train.__file__).parent.glob("*.py"))}
        train.verify_prefetch_source_change(args.run, old["source_sha256"], sources)
        if (sha256(args.study) != old["study_sha256"] or sha256(args.cache / "manifest.json") != old["manifest_sha256"]
                or sha256(args.fast_cache / "manifest.json") != old["fast_manifest_sha256"]
                or payload["provenance"] != {**old, "branch": "M0"}
                or config != train.branch_config(plan["study"], "M0", False, config["seed"])):
            raise ValueError("Checkpoint, study or cache identity mismatch")
        if args.dry_run:
            print(json.dumps({"status": "throughput_plan_valid", "checkpoint_update": payload["updates"],
                              "seed": config["seed"], "measured_attempts_per_pass": args.steps,
                              "order": ["serial", "prefetch", "prefetch", "serial"], "output": str(args.output)}))
            return
        if args.output.exists():
            raise FileExistsError("Benchmark output exists; choose another --output")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        args.output.mkdir(parents=True)
        torch.set_num_threads(config["cpu_threads"])
        blosc2.set_nthreads(config["cpu_threads"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        cases = sorted(plan["split"]["partitions"]["M0_train"])
        ignore = len(plan["labels"]) + 1
        result = {"status": "running", "purpose": "throughput_and_input_equivalence_not_research_training",
                  "run": str(args.run.resolve()), "checkpoint_sha256": digest, "checkpoint_update": payload["updates"],
                  "seed": config["seed"], "source_sha256": sources, "original_provenance": old,
                  "source_snapshot_sha256": train.save_source_snapshot(args.output),
                  "passes": [], "hardware_probes": [], "error": None}
        atomic_json(args.output / "result.json", result)

        def event(value):
            print(json.dumps(value), flush=True)
            with (args.output / "run.log").open("a") as stream:
                stream.write(json.dumps(value) + "\n")

        def new_loader():
            return train.CaseCache(args.cache, manifest, normalization, cases, config["case_cache_gib"], args.fast_cache, fast)

        def restore_rng(rng):
            import copy
            rng.bit_generator.state = copy.deepcopy(payload["numpy_rng"])
            np.random.set_state(payload["numpy_global_rng"])
            random.setstate(payload["python_rng"])
            torch.set_rng_state(payload["torch_rng"])
            torch.cuda.set_rng_state_all(payload["cuda_rng"])

        model = None
        try:
            traces = []
            # Hash the same augmented inputs separately from timed passes, avoiding timing distortion.
            for use_prefetch in (False, True):
                rng = np.random.default_rng()
                restore_rng(rng)
                loader = new_loader()
                reader = train.CasePrefetcher(loader) if use_prefetch else loader
                trace = hashlib.sha256()
                try:
                    for index in range(16):
                        case, x, y = train.training_input(reader, cases, config, rng, ignore)
                        if use_prefetch and index < 15:
                            reader.prefetch(cases, rng)
                        trace.update(case.encode())
                        trace.update(x.cpu().numpy().tobytes())
                        trace.update(y.cpu().numpy().tobytes())
                        trace.update(json.dumps(rng.bit_generator.state, sort_keys=True).encode())
                        trace.update(torch.cuda.get_rng_state().cpu().numpy().tobytes())
                        del x, y
                    traces.append(trace.hexdigest())
                finally:
                    if use_prefetch:
                        reader.close()
                    loader.clear()
            if traces[0] != traces[1]:
                raise ValueError("Prefetch changes actual case/patch/augmentation/RNG sequence")
            result["input_equivalence"] = {"cases": 16, "equal": True, "trace_sha256": traces[0]}
            event({"input_equivalence": result["input_equivalence"]})

            model = train.build_model(config, ignore)
            first_update = None
            for use_prefetch in (False, True, True, False):
                mode = "prefetch" if use_prefetch else "serial"
                loader = new_loader()
                reader = train.CasePrefetcher(loader) if use_prefetch else loader
                rng = np.random.default_rng()
                optimizer, loss_fn, scaler = train.optimization_components(model, config, ignore)
                initial_update = train.load_checkpoint(checkpoint, model, optimizer, scaler, rng, config, payload["provenance"])
                successful, skipped_count = 0, 0
                waits, computes, totals, trace_cases = [], [], [], []
                torch.cuda.reset_peak_memory_stats()
                try:
                    for index in range(args.steps + 10):
                        torch.cuda.synchronize()
                        begin = time.perf_counter()
                        case, x, y = train.training_input(reader, cases, config, rng, ignore)
                        if use_prefetch and index + 1 < args.steps + 10:
                            reader.prefetch(cases, rng)
                        ready = time.perf_counter()
                        for group in optimizer.param_groups:
                            group["lr"] = train.learning_rate_at(config, initial_update + successful)
                        loss, skipped = train.update_step(model, x, y, optimizer, loss_fn, scaler)
                        torch.cuda.synchronize()
                        end = time.perf_counter()
                        successful += int(not skipped)
                        skipped_count += int(skipped)
                        trace_cases.append(case)
                        del x, y
                        if index == 0:
                            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                            if first_update is None:
                                first_update = state
                            else:
                                for k in state:
                                    torch.testing.assert_close(state[k], first_update[k], rtol=1e-5, atol=1e-6)
                            del state
                        if index >= 10:
                            waits.append(ready - begin)
                            computes.append(end - ready)
                            totals.append(end - begin)
                    row = {"mode": mode, "measured_attempts": args.steps, "mean_seconds_per_update_attempt": float(np.mean(totals)),
                           "mean_input_wait_and_augmentation_seconds": float(np.mean(waits)),
                           "mean_compute_seconds": float(np.mean(computes)), "amp_skips": skipped_count,
                           "case_sequence_sha256": hashlib.sha256(json.dumps(trace_cases).encode()).hexdigest(),
                           "peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
                           "peak_reserved_MiB": torch.cuda.max_memory_reserved() / 2**20}
                    result["passes"].append(row)
                    event(row)
                    atomic_json(args.output / "result.json", result)
                finally:
                    if use_prefetch:
                        reader.close()
                    loader.clear()
                    del optimizer, loss_fn, scaler
                    torch.cuda.empty_cache()
            if len({p["case_sequence_sha256"] for p in result["passes"]}) != 1:
                raise ValueError("Timed passes sampled different case sequences")
            result["one_update_weights_close"] = True
            serial = np.mean([p["mean_seconds_per_update_attempt"] for p in result["passes"] if p["mode"] == "serial"])
            prefetched = np.mean([p["mean_seconds_per_update_attempt"] for p in result["passes"] if p["mode"] == "prefetch"])
            result["prefetch_speedup"] = float(serial / prefetched)
            result["prefetch_recommended"] = bool(serial / prefetched >= 1.05 and not any(p["amp_skips"] for p in result["passes"]))
            # VRAM/compute trials use cached training patches; changes are not applied to the research protocol.
            loader = new_loader()
            data, seg, locations = loader.get(cases[len(cases) // 2])
            for batch, patch_size in ((1, config["patch_size"]), (2, config["patch_size"]),
                                      (4, config["patch_size"]), (2, [128, 128, 128])):
                x = y = parts = optimizer = loss_fn = scaler = None
                row = {"batch_size": batch, "patch_size": patch_size, "scope": "resident_GPU_compute_only_not_end_to_end",
                       "changes_research_protocol": batch != 1 or patch_size != config["patch_size"]}
                try:
                    rng = np.random.default_rng(config["seed"])
                    local = {**config, "patch_size": patch_size}
                    parts = [train.sample_patch(data, seg, local, rng, ignore, device="cpu", locations=locations) for _ in range(batch)]
                    x = torch.cat([p[0] for p in parts]).cuda()
                    y = torch.cat([p[1] for p in parts]).cuda()
                    parts = None
                    optimizer, loss_fn, scaler = train.optimization_components(model, config, ignore)
                    train.load_checkpoint(checkpoint, model, optimizer, scaler, rng, config, payload["provenance"])
                    torch.cuda.reset_peak_memory_stats()
                    times, skips = [], 0
                    for index in range(8):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        _, skipped = train.update_step(model, x, y, optimizer, loss_fn, scaler)
                        torch.cuda.synchronize()
                        if index >= 3:
                            times.append(time.perf_counter() - start)
                        skips += int(skipped)
                    row.update(status="complete", seconds_per_batch=float(np.mean(times)),
                               patches_per_second=float(batch / np.mean(times)), amp_skips=skips,
                               peak_allocated_MiB=torch.cuda.max_memory_allocated() / 2**20,
                               peak_reserved_MiB=torch.cuda.max_memory_reserved() / 2**20)
                except torch.OutOfMemoryError:
                    row["status"] = "CUDA_OOM"
                finally:
                    model.zero_grad(set_to_none=True)
                    del x, y, parts, optimizer, loss_fn, scaler
                    torch.cuda.empty_cache()
                result["hardware_probes"].append(row)
                event(row)
                atomic_json(args.output / "result.json", result)
            loader.clear()
            if sha256(checkpoint) != digest:
                raise ValueError("Research checkpoint changed during benchmark")
            result["status"] = "throughput_benchmark_complete"
            result["process_peak_RSS_MiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except BaseException as exc:
            result.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                          error=type(exc).__name__ + ": " + str(exc))
            raise
        finally:
            atomic_json(args.output / "result.json", result)
            if model is not None:
                del model
            torch.cuda.empty_cache()
        event({"status": result["status"], "prefetch_speedup": result["prefetch_speedup"],
               "prefetch_recommended": result["prefetch_recommended"], "result": str(args.output / "result.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--splits", type=Path, default=ROOT / "manifests/splits_v300/splits.json")
    parser.add_argument("--qc", type=Path, default=ROOT / "manifests/qc_v300/case_qc.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/probe.json")
    parser.add_argument("--study", type=Path, help="Use selected study labels and review-screened development cases for a mechanical probe only")
    parser.add_argument("--review", type=Path, default=ROOT / "manifests/review_v300")
    parser.add_argument("--dry-run", action="store_true", help="Validate selection/config and print the plan without loading raw voxels or training")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/feasibility_v300")
    parser.add_argument("--throughput", action="store_true", help="Benchmark a paused training run without modifying its checkpoints")
    parser.add_argument("--run", type=Path, help="Paused research run for --throughput")
    parser.add_argument("--steps", type=int, default=100, help="Measured attempts in each serial/prefetch pass")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/preprocessed/abdominal_v1")
    parser.add_argument("--fast-cache", type=Path, default=ROOT / "data/preprocessed/abdominal_v1_fast")
    args = parser.parse_args()
    if args.throughput:
        throughput_probe(args)
        return
    if args.study and args.output == ROOT / "outputs/feasibility_v300":
        args.output = ROOT / "outputs/feasibility_abdominal_v1"
    if args.output.exists() and not args.dry_run:
        raise FileExistsError("Probe output exists; choose a new run directory")
    if not args.dry_run and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    _, _, labels, protocol, input_provenance = load_inputs(args.raw, args.inventory, args.protocol)
    split = json.loads(args.splits.read_text())
    rows = read_csv(args.qc)
    if split["provenance"]["case_qc_sha256"] != sha256(args.qc) or split["provenance"]["protocol_sha256"] != sha256(args.protocol):
        raise ValueError("QC/split/config provenance mismatch")
    validate_splits(split["partitions"], rows, protocol)
    config = json.loads(args.config.read_text())
    allowed = None
    extra_provenance = {}
    if args.study:
        from .review import load_review_ledger, panel_labels
        study = json.loads(args.study.read_text())
        if sha256(ROOT / study["qc_protocol"]) != sha256(args.protocol):
            raise ValueError("Study and probe QC protocols differ")
        _, screened, reviews, _ = load_review_ledger(study, args.review, args.qc.parent)
        selected_labels = panel_labels(study)
        if not set(selected_labels) <= set(labels):
            raise ValueError("Selected labels missing from inventory")
        labels = selected_labels
        config.update(study["training"]["feasibility_probe_overrides"])
        config["seed"] = study["training"]["paired_training_seeds"][0]
        allowed = {case for case, row in screened.items() if int(row["nonempty_no_border_organs"]) == len(labels)
                   and not any(r["annotation"] == "reject" or r["coverage"] in {"partial", "out_of_fov"} for r in reviews[case])}
        extra_provenance = {"study_sha256": sha256(args.study), "review_csv_sha256": sha256(args.review / "review.csv"),
                            "review_cases_sha256": sha256(args.review / "cases.csv"),
                            "selection_scope": "historical_disjoint_development_roles_filtered_for_mechanical_probe_not_final_study_split"}
    if config["batch_size"] != 1:
        raise ValueError("This bounded probe implements batch_size=1 only")
    divisor = 2 ** (len(config["features_per_stage"]) - 1)
    if any(p % divisor for p in config["patch_size"]):
        raise ValueError("Patch dimensions must match the network downsampling stages")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    by_id = {r["case_id"]: r for r in rows}
    def representatives(role, two=False):
        candidates = sorted((case for case in split["partitions"][role] if allowed is None or case in allowed),
                            key=lambda x: (np.prod(json.loads(by_id[x]["shape"])), x))
        if len(candidates) < (2 if two else 1):
            raise ValueError("Empty development pool: " + role)
        return [candidates[min(len(candidates)//2, len(candidates)-2)], candidates[-1]] if two else [candidates[-1]]
    selected = {role: representatives(role, two=True) for role in ("M0_train", "M1_finetune", "M2_finetune")}
    validation = representatives("adult_validation") + representatives("pediatric_validation")
    selected_ids = sorted({x for cases in selected.values() for x in cases} | set(validation))
    if any(by_id[x]["official_split"] == "test" for x in selected_ids):
        raise ValueError("Test case entered feasibility tuning")
    prepared_estimate = 0
    for case_id in selected_ids:
        row = by_id[case_id]
        shape, spacing = np.array(json.loads(row["shape"])), np.array(json.loads(row["spacing"]))
        if args.study and not np.allclose(spacing, config["target_spacing"], rtol=0, atol=1e-4):
            raise ValueError("Native-grid probe requires all selected source spacings to match the target")
        target_shape = np.maximum(1, np.round(shape * spacing / config["target_spacing"])).astype(int)
        prepared_estimate += int(np.prod(target_shape)) * 5
    if prepared_estimate / 2**30 > config.get("max_prepared_arrays_gib", float("inf")):
        raise MemoryError("Prepared array estimate exceeds the configured host-memory guard")
    if args.dry_run:
        print(json.dumps({"status": "plan_validated_no_voxel_loading_no_training", "selected_cases": selected,
                          "validation_cases": validation, "labels": labels, "config": config,
                          "prepared_arrays_estimate_gib": prepared_estimate / 2**30,
                          "memory_note": "Array estimate only; peak RSS also includes decoding, model states and inference logits.",
                          "output": str(args.output)}, ensure_ascii=False, indent=2))
        return
    args.output.mkdir(parents=True)
    def event(value):
        line = json.dumps(value, ensure_ascii=False)
        print(line, flush=True)
        with (args.output / "run.log").open("a", encoding="utf-8") as log:
            log.write(line + "\n")
    save_artifacts = config.get("save_artifacts", True)
    result = {"status": "running", "purpose": config["purpose"], "selected_cases": selected,
              "validation_cases": validation, "config": config,
              "versions": {p: importlib.metadata.version(p) for p in ("torch", "numpy", "nibabel", "scipy", "nnunetv2", "dynamic-network-architectures")},
              "gpu": torch.cuda.get_device_name(), "gpu_total_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
              "environment": {"python": sys.version, "platform": platform.platform(),
                              "gpu_free_before_mib": torch.cuda.mem_get_info()[0] / 2**20,
                              "driver_version": subprocess.check_output([
                                  "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()},
              "provenance": {**input_provenance, **extra_provenance, **git_provenance(), "split_sha256": sha256(args.splits),
                             "probe_config_sha256": sha256(args.config), "probe_code_sha256": sha256(__file__),
                             "training_code_sha256": sha256(Path(__file__).with_name("training.py")),
                             "preprocessing_code_sha256": sha256(Path(__file__).with_name("preprocessing.py")),
                             "uv_lock_sha256": sha256(ROOT / "uv.lock")},
              "preprocessing": [], "training": {}, "inference": {}}
    result["prepared_arrays_estimate_gib"] = prepared_estimate / 2**30
    result["artifact_policy"] = "save_arrays_checkpoints_predictions" if save_artifacts else "measurements_and_log_only"
    write_json(args.output / "result.json", result)
    start_all = time.perf_counter()
    try:
        prepared, properties, references, samples = {}, {}, {}, []
        for case_id in selected_ids:
            data, seg, sample, props, reference = preprocess_case(args.raw, case_id, labels, config["target_spacing"])
            prepared[case_id] = (data, seg)
            properties[case_id], references[case_id] = props, reference
            result["preprocessing"].append(props)
            if case_id in selected["M0_train"]:
                samples.append(sample)
            write_json(args.output / "result.json", result)
            event({"preprocessed": case_id, "shape": props["preprocessed_shape"], "seconds": props["preprocessing_seconds"]})
        normalization = fit_normalization(samples, selected["M0_train"])
        result["normalization"] = normalization
        for case_id, (data, seg) in prepared.items():
            normalize_inplace(data, normalization)
            if save_artifacts:
                np.savez_compressed(args.output / (case_id + "_preprocessed.npz"), data=data, seg=seg)
        del samples
        model = build_model(config, len(labels) + 1)
        result["parameter_count"] = sum(p.numel() for p in model.parameters())
        result["labels"] = {"background": 0, **{name: i for i, name in enumerate(labels, 1)}, "ignore_overlap": len(labels)+1}
        m0_state = None
        for name, role in (("probe_M0", "M0_train"), ("probe_M1", "M1_finetune"), ("probe_M2", "M2_finetune")):
            if m0_state is not None:
                model.load_state_dict(m0_state)
            initial_hash = state_hash(model.state_dict())
            measured = train_interval(model, selected[role], prepared, config, len(labels)+1)
            measured["initial_state_sha256"] = initial_hash
            measured["weights_changed"] = state_hash(model.state_dict()) != initial_hash
            if measured["skipped_amp_steps"] or not measured["weights_changed"]:
                raise RuntimeError("Probe did not complete every optimizer update")
            result["training"][name] = measured
            if name == "probe_M0":
                m0_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if save_artifacts:
                torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                            "config": config, "purpose": config["purpose"]}, args.output / (name + ".pt"))
            result["inference"][name] = []
            for case_id in validation:
                prediction, timings = infer_volume(model, prepared[case_id][0], labels, config)
                props = properties[case_id]
                restored = resample_data_or_seg_to_shape(prediction[None], props["original_shape"], config["target_spacing"],
                                                         props["original_spacing"], is_seg=True, order=0, force_separate_z=False)[0]
                reference = references[case_id]
                header = reference.header.copy()
                header.set_data_dtype(np.uint8)
                if tuple(restored.shape) != tuple(reference.shape) or restored.min() < 0 or restored.max() > len(labels):
                    raise ValueError("Restored prediction grid or label range is invalid")
                timings["restored_shape_validated"] = True
                if save_artifacts:
                    nib.save(nib.Nifti1Image(restored.astype(np.uint8), reference.affine, header),
                             args.output / (name + "_" + case_id + "_prediction.nii.gz"))
                timings["case_id"] = case_id
                result["inference"][name].append(timings)
                del prediction, restored
            write_json(args.output / "result.json", result)
            event({"branch_complete": name, "training": measured})
        if result["training"]["probe_M1"]["initial_state_sha256"] != result["training"]["probe_M2"]["initial_state_sha256"]:
            raise AssertionError("M1/M2 initialization differs")
        result["status"] = "mechanical_probe_passed_research_gates_pending"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = type(exc).__name__ + ":" + str(exc)
        raise
    finally:
        result["elapsed_seconds"] = time.perf_counter() - start_all
        result["peak_process_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        result["artifact_bytes"] = sum(p.stat().st_size for p in args.output.iterdir() if p.is_file())
        result["limitations"] = ["Short updates are not trained M0/M1/M2 models or accuracy evidence.",
            "Architecture is an explicitly configured PlainConvUNet, not an nnU-Net auto-planned benchmark.",
            "Configured spacing and overlapping-label ignore policy require scientific review.",
            "Full inference was tested on selected development volumes only; no held-out test was used.",
            "Unknown NIfTI spatial units and patient mapping remain unresolved."]
        write_json(args.output / "result.json", result)
        event({"status": result["status"], "result": str(args.output / "result.json"), "error": result.get("error")})


if __name__ == "__main__":
    main()
