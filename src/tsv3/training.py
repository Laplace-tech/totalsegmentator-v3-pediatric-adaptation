"""공유 학습 구성 요소와 고정 development cache 기반 실행기."""

import hashlib
import json
import os
import random
import time
from pathlib import Path
from collections import OrderedDict
from datetime import datetime, timezone

os.environ.setdefault("nnUNet_compile", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from .evaluation import validation_dice


def state_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def sample_patch(data, seg, config, rng, ignore, device="cuda", locations=None):
    patch = np.array(config["patch_size"])
    shape = np.array(data.shape[1:])
    if rng.random() < config["foreground_sampling_probability"]:
        if locations is not None:
            organ = rng.choice(np.unique(locations[:, 0]))
            candidates = locations[locations[:, 0] == organ, 1:]
            center = candidates[rng.integers(len(candidates))]
        else:
            foreground = np.flatnonzero((seg[0] > 0) & (seg[0] != ignore))
            center = np.array(np.unravel_index(rng.choice(foreground), shape)) if foreground.size else shape // 2
        start = np.clip(center - patch // 2, 0, np.maximum(shape - patch, 0))
    else:
        start = np.array([rng.integers(0, max(1, int(n-p+1))) for n, p in zip(shape, patch)])
    slices = (slice(None),) + tuple(slice(int(s), int(s+p)) for s, p in zip(start, patch))
    x, y = data[slices], seg[slices]
    padding = [(0, 0)] + [(0, max(0, int(p-n))) for n, p in zip(x.shape[1:], patch)]
    x = np.pad(x, padding, constant_values=0)
    y = np.pad(y, padding, constant_values=ignore)
    return torch.from_numpy(x[None].copy()).to(device), torch.from_numpy(y[None].astype(np.int64)).to(device)


def architecture(config):
    stages = len(config["features_per_stage"])
    return {"network_class_name": "dynamic_network_architectures.architectures.unet.PlainConvUNet",
            "arch_kwargs": {"n_stages": stages, "features_per_stage": config["features_per_stage"],
                            "conv_op": "torch.nn.Conv3d", "kernel_sizes": [[3, 3, 3]] * stages,
                            "strides": [[1, 1, 1]] + [[2, 2, 2]] * (stages-1),
                            "n_conv_per_stage": [2] * stages, "n_conv_per_stage_decoder": [2] * (stages-1),
                            "conv_bias": True, "norm_op": "torch.nn.InstanceNorm3d",
                            "norm_op_kwargs": {"eps": 1e-5, "affine": True},
                            "nonlin": "torch.nn.LeakyReLU", "nonlin_kwargs": {"negative_slope": 0.01, "inplace": True}},
            "_kw_requires_import": ["conv_op", "norm_op", "nonlin"]}


def build_model(config, classes):
    spec = architecture(config)
    return get_network_from_plans(spec["network_class_name"], spec["arch_kwargs"], spec["_kw_requires_import"],
                                  1, classes, deep_supervision=False).cuda()


def optimization_components(model, config, ignore, device="cuda"):
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=config["momentum"],
                                weight_decay=config["weight_decay"], nesterov=True)
    loss_fn = DC_and_CE_loss({"batch_dice": True, "smooth": 1e-5, "do_bg": False, "ddp": False}, {},
                             ignore_label=ignore, dice_class=MemoryEfficientSoftDiceLoss)
    scaler = torch.amp.GradScaler(device, init_scale=1024, enabled=device == "cuda")
    return optimizer, loss_fn, scaler


def update_step(model, x, y, optimizer, loss_fn, scaler):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(x.device.type, dtype=torch.float16, enabled=x.is_cuda):
        loss = loss_fn(model(x), y)
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite training loss")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), 12)
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach()), scaler.get_scale() < old_scale


def save_checkpoint(path, model, optimizer, scaler, rng, updates, config, provenance, trainer_state=None):
    """자체 생성 checkpoint만 읽는다. 모델·optimizer·AMP·RNG·입력 identity를 함께 저장한다."""
    payload = {"format_version": 1, "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "updates": updates,
               "config": config, "provenance": provenance, "numpy_rng": rng.bit_generator.state,
               "numpy_global_rng": np.random.get_state(),
               "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
               "cuda_rng": torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else None,
               "trainer_state": trainer_state}
    temporary = Path(str(path) + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path, model, optimizer, scaler, rng, config, provenance):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["format_version"] != 1 or payload["config"] != config or payload["provenance"] != provenance:
        raise ValueError("Checkpoint configuration or input provenance changed")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scaler.load_state_dict(payload["scaler"])
    rng.bit_generator.state = payload["numpy_rng"]
    np.random.set_state(payload["numpy_global_rng"])
    random.setstate(payload["python_rng"])
    torch.set_rng_state(payload["torch_rng"])
    if payload["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return payload["updates"]


def train_interval(model, cases, prepared, config, ignore):
    model.train()
    rng = np.random.default_rng(config["seed"])
    optimizer, loss_fn, scaler = optimization_components(model, config, ignore)
    torch.cuda.reset_peak_memory_stats()
    times, losses = [], []
    skipped_steps = 0
    start_all = time.perf_counter()
    for step in range(config["warmup_steps"] + config["measured_steps"]):
        torch.cuda.synchronize()
        start = time.perf_counter()
        case_id = cases[step % len(cases)]
        data, seg = prepared[case_id]
        x, y = sample_patch(data, seg, config, rng, ignore)
        loss, skipped = update_step(model, x, y, optimizer, loss_fn, scaler)
        skipped_steps += int(skipped)
        torch.cuda.synchronize()
        if step >= config["warmup_steps"]:
            times.append(time.perf_counter() - start)
            losses.append(loss)
        del x, y, loss
    result = {"total_steps": config["warmup_steps"] + config["measured_steps"],
              "measured_steps": len(times), "skipped_amp_steps": skipped_steps,
              "optimizer_updates": config["warmup_steps"] + config["measured_steps"] - skipped_steps,
              "median_step_seconds": float(np.median(times)),
              "step_seconds": times, "finite_loss": True, "losses_not_accuracy_results": losses,
              "wall_seconds": time.perf_counter() - start_all,
              "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
              "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
    del optimizer, scaler, loss_fn
    torch.cuda.empty_cache()
    return result


def infer_volume(model, data, labels, config):
    estimated = int(np.prod(data.shape[1:])) * (len(labels) + 1) * 2 / 2**30
    if estimated > config["max_inference_logits_gib"]:
        raise MemoryError(f"Host logit allocation {estimated:.2f} GiB exceeds probe limit")
    plans = PlansManager({"configurations": {"probe": {"patch_size": config["patch_size"], "architecture": architecture(config)}}})
    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                               perform_everything_on_device=False, device=torch.device("cuda"), allow_tqdm=False)
    dataset = {"labels": {"background": 0, **{name: i for i, name in enumerate(labels, 1)}}}
    predictor.manual_initialization(model, plans, plans.get_configuration("probe"), None, dataset, "mechanical_probe", None)
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    logits = predictor.predict_sliding_window_return_logits(torch.from_numpy(data))
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite inference logits")
    prediction = logits.argmax(0).numpy().astype(np.uint8)
    torch.cuda.synchronize()
    result = {"full_preprocessed_volume": True, "finite_logits": True, "shape": list(prediction.shape),
              "seconds": time.perf_counter() - start,
              "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
              "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
              "host_logits_estimate_gib": estimated}
    del logits, predictor
    torch.cuda.empty_cache()
    return prediction, result


def augment_patch(x, y, config, rng, ignore):
    """정규화된 intensity만 변환한다. 좌우 label이나 geometry를 바꾸지 않는다."""
    if rng.random() < config["brightness_probability"]:
        x = x * float(rng.uniform(*config["brightness_range"]))
    if rng.random() < config["contrast_probability"]:
        valid = y != ignore
        mean = x[valid].mean() if valid.any() else x.new_zeros(())
        x = (x - mean) * float(rng.uniform(*config["contrast_range"])) + mean
    if rng.random() < config["noise_probability"]:
        x = x + torch.randn_like(x) * float(rng.uniform(*config["noise_std_range"]))
    return x.masked_fill(y == ignore, 0), y


class CaseCache:
    """공유 NPZ를 bounded LRU로 읽는다. RAM cache는 표본 추출 분포에 관여하지 않는다."""

    def __init__(self, root, manifest, normalization, allowed, limit_gib, fast_root=None, fast_manifest=None):
        self.root, self.manifest, self.normalization = root, manifest, normalization
        self.fast_root, self.fast_manifest = fast_root, fast_manifest
        self.allowed = set(allowed)
        self.limit = int(limit_gib * 2**30)
        self.items = OrderedDict()
        self.bytes = 0
        self.verified = set()

    def clear(self):
        self.items.clear()
        self.bytes = 0

    def get(self, case):
        from .data import sha256
        from .preprocessing import normalize_inplace
        if case not in self.allowed:
            raise ValueError("Case outside authorized development roles: " + case)
        if case in self.items:
            self.items.move_to_end(case)
            return self.items[case]
        record = (self.fast_manifest or self.manifest)["completed"][case]
        root = self.fast_root or self.root
        path = root / record["file"]
        if path.parent.resolve() != root.resolve() or path.name != case + ".npz":
            raise ValueError("Unexpected cache path")
        original = self.manifest["completed"][case]
        expected_bytes = int(np.prod(original["properties"]["preprocessed_shape"])) * 5 + 7 * 10000 * 16
        if expected_bytes > self.limit:
            raise MemoryError("Case exceeds RAM cache allowance: " + case)
        while self.items and self.bytes + expected_bytes > self.limit:
            _, previous = self.items.popitem(last=False)
            self.bytes -= sum(a.nbytes for a in previous)
            del previous
        if case not in self.verified:
            if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
                raise ValueError("Cache file changed: " + case)
            self.verified.add(case)
        if self.fast_root is None:
            with np.load(path, allow_pickle=False) as arrays:
                data, seg, locations = arrays["data"], arrays["seg"], arrays["foreground_locations"]
        else:
            from .fast_cache import read_case
            data, seg, locations = read_case(path)
        if data.dtype != np.float32 or seg.dtype != np.uint8 or data.shape != seg.shape:
            raise ValueError("Invalid cached arrays: " + case)
        normalize_inplace(data, self.normalization)
        value = (data, seg, locations)
        self.bytes += sum(a.nbytes for a in value)
        self.items[case] = value
        return value


def learning_rate_at(config, updates):
    total = config["optimizer_updates"]
    if not 0 <= updates <= total:
        raise ValueError("Update outside fixed learning-rate schedule")
    return config["learning_rate"] * (1 - updates / total) ** config["schedule"]["power"]


class CasePrefetcher:
    """One CPU read ahead; lookahead never advances the authoritative training RNG."""

    def __init__(self, loader):
        from concurrent.futures import ThreadPoolExecutor
        self.loader = loader
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ct-prefetch")
        self.pending = None
        self.expected_case = None

    def get(self, case):
        if self.pending is None:
            return self.loader.get(case)
        if case != self.expected_case:
            raise ValueError("Prefetched case differs from authoritative RNG sequence")
        result = self.pending.result()
        self.pending = None
        self.expected_case = None
        return result

    def prefetch(self, cases, rng):
        import copy
        if self.pending is not None:
            raise RuntimeError("Only one pending case is permitted")
        lookahead = np.random.default_rng()
        lookahead.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
        self.expected_case = cases[int(lookahead.integers(len(cases)))]
        self.pending = self.executor.submit(self.loader.get, self.expected_case)

    def drain(self):
        if self.pending is not None:
            self.pending.result()
            self.pending = None
            self.expected_case = None

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.pending = None


def training_input(loader, cases, config, rng, ignore, device="cuda"):
    case = cases[int(rng.integers(len(cases)))]
    data, seg, locations = loader.get(case)
    x, y = sample_patch(data, seg, config, rng, ignore, device=device, locations=locations)
    del data, seg, locations
    x, y = augment_patch(x, y, config["augmentation"], rng, ignore)
    return case, x, y


def validate_model(model, loader, partitions, roles, labels, config):
    loader.clear()
    results = {}
    for role in roles:
        scores, timings = {}, {}
        for case in sorted(partitions[role]):
            data, seg, _ = loader.get(case)
            prediction, timing = infer_volume(model, data, labels, config)
            scores[case] = validation_dice(prediction, seg[0], len(labels))
            timings[case] = timing["seconds"]
            del data, seg, prediction
            loader.clear()
        results[role] = {"cases": scores, "organ_mean_Dice": np.mean(list(scores.values()), axis=0).tolist(),
                         "macro_Dice": float(np.mean(list(scores.values()))), "inference_seconds": timings}
    return results


def training_seed(study, seed=None):
    baseline = study["training"]["paired_training_seeds"][0]
    allowed = study["training"].get("replication_plan", {}).get("seeds", [baseline])
    selected = baseline if seed is None else seed
    if type(selected) is not int or not 0 <= selected < 2**32 or selected not in allowed:
        raise ValueError(f"Training seed must be one of the declared seeds: {allowed}")
    return selected


def training_output(study, seed, smoke=False, explicit_seed=False):
    from .data import ROOT
    stem = "training_smoke" if smoke else "training"
    suffix = "" if smoke and not explicit_seed else f"_seed{seed}"
    return ROOT / "outputs" / f"{stem}_{study['study_version']}{suffix}"


def check_output_seed(output, seed):
    path = output / "run.json"
    if path.exists():
        run = json.loads(path.read_text())
        previous_seed = run["provenance"].get("training_seed")
        if previous_seed is None:
            previous_seed = run["study_snapshot"]["training"]["paired_training_seeds"][0]
        if previous_seed != seed:
            raise ValueError(f"Output belongs to seed {previous_seed}, requested {seed}: {output}")


def branch_config(study, branch, smoke, seed=None):
    import copy
    numeric = copy.deepcopy(study["training"]["numeric_training_plan"])
    config = {**numeric, **numeric[branch], **study["training"]["fixed_execution_plan"],
              "seed": training_seed(study, seed), "branch": branch, "smoke": smoke}
    if smoke:
        config.update(optimizer_updates=numeric["smoke"]["updates_per_branch"],
                      validation_every_updates=numeric["smoke"]["validation_every_updates"],
                      checkpoint_every_updates=numeric["smoke"]["checkpoint_every_updates"])
    return config


def run_branch(branch, plan, cache, output, config, provenance, m0_checkpoint, resume, host_volume,
               fast_cache=None, fast_manifest=None, legacy_m0=None, prefetch=False):
    from .data import sha256
    from .preprocessing import atomic_json, check_space
    from .night import verify_completion
    manifest, normalization = verify_completion(cache, plan)
    directory = output / branch
    directory.mkdir(exist_ok=True)
    latest, selected = directory / "latest.pt", directory / "selected.pt"
    if latest.exists() and not resume:
        raise FileExistsError("Existing branch checkpoint; explicitly use --resume")
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    model = build_model(config, len(plan["labels"]) + 1)
    branch_provenance = {**provenance, "branch": branch}
    if branch != "M0":
        source = torch.load(m0_checkpoint, map_location="cpu", weights_only=False)
        if source["provenance"] != {**provenance, "branch": "M0"} or source["config"] != branch_config(plan["study"], "M0", config["smoke"], config["seed"]):
            raise ValueError("M0 checkpoint belongs to a different experiment")
        model.load_state_dict(source["model"])
        branch_provenance["M0_checkpoint_sha256"] = sha256(m0_checkpoint)
        branch_provenance["M0_state_sha256"] = state_hash(source["model"])
        if state_hash(model.state_dict()) != branch_provenance["M0_state_sha256"]:
            raise ValueError("Branch initialization differs from selected M0")
        del source
    optimizer, loss_fn, scaler = optimization_components(model, config, len(plan["labels"]) + 1)
    state = {"initial_state_sha256": state_hash(model.state_dict()), "best_adult_Dice": None,
             "best_update": None, "amp_skips": 0, "attempts": 0, "training_seconds": 0.0,
             "validation_seconds": 0.0, "last_validation_update": 0}
    updates = 0
    if resume and latest.exists():
        checkpoint_provenance = branch_provenance
        if branch == "M0" and legacy_m0 and sha256(latest) == legacy_m0["checkpoint_sha256"]:
            checkpoint_provenance = {**legacy_m0["provenance"], "branch": "M0"}
        updates = load_checkpoint(latest, model, optimizer, scaler, rng, config, checkpoint_provenance)
        state = torch.load(latest, map_location="cpu", weights_only=False)["trainer_state"]
        if state is None:
            raise ValueError("Missing trainer state in checkpoint")
    partitions = plan["split"]["partitions"]
    train_role = {"M0": "M0_train", "M1": "M1_finetune", "M2": "M2_finetune"}[branch]
    cases = sorted(partitions[train_role])
    validation_roles = ["adult_validation", "pediatric_validation"] if config["smoke"] else config["validation_roles"][branch]
    allowed = cases + [c for role in validation_roles for c in partitions[role]]
    loader = CaseCache(cache, manifest, normalization, allowed, config["case_cache_gib"], fast_cache, fast_manifest)
    reader = CasePrefetcher(loader) if prefetch else loader
    consecutive_skips = 0
    total = config["optimizer_updates"]
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    def checkpoint(path):
        check_space(directory, host_volume, 512 * 2**20, config["disk_reserve_gib"])
        save_checkpoint(path, model, optimizer, scaler, rng, updates, config, branch_provenance, state)

    def report(status, **extra):
        value = {"status": status, "branch": branch, "seed": config["seed"], "updates": updates, "budget": total,
                 "smoke": config["smoke"], "updated_utc": datetime.now(timezone.utc).isoformat(),
                 "trainer_state": state.copy(), **extra}
        atomic_json(directory / "status.json", value)
        with (output / "run.log").open("a") as stream:
            stream.write(json.dumps(value) + "\n")
        print(json.dumps(value), flush=True)

    try:
        report("running")
        while updates < total:
            tick = time.monotonic()
            case, x, y = training_input(reader, cases, config, rng, len(plan["labels"]) + 1)
            if prefetch and updates + 1 < total:
                reader.prefetch(cases, rng)
            lr = learning_rate_at(config, updates)
            for group in optimizer.param_groups:
                group["lr"] = lr
            loss, skipped = update_step(model, x, y, optimizer, loss_fn, scaler)
            torch.cuda.synchronize()
            del x, y
            state["training_seconds"] += time.monotonic() - tick
            state["attempts"] += 1
            if skipped:
                state["amp_skips"] += 1
                consecutive_skips += 1
                if consecutive_skips >= config["max_consecutive_amp_skips"] or state["amp_skips"] >= config["max_total_amp_skips"]:
                    raise RuntimeError("AMP skip limit reached; budget cannot be silently shortened")
                continue
            updates += 1
            consecutive_skips = 0
            if updates % config["log_every_updates"] == 0:
                report("running", loss=loss, learning_rate=lr)
            if updates % config["validation_every_updates"] == 0 or updates == total:
                if prefetch:
                    reader.drain()
                tick = time.monotonic()
                validation = validate_model(model, loader, partitions, validation_roles, plan["labels"], config)
                state["validation_seconds"] += time.monotonic() - tick
                state["last_validation_update"] = updates
                atomic_json(directory / f"validation_{updates:06d}.json", validation)
                score = validation["adult_validation"]["macro_Dice"]
                if branch == "M0" and (state["best_adult_Dice"] is None or score > state["best_adult_Dice"]):
                    state.update(best_adult_Dice=score, best_update=updates)
                    checkpoint(selected)
                report("validation_complete", validation_macro_Dice={k: v["macro_Dice"] for k, v in validation.items()})
            if updates % config["checkpoint_every_updates"] == 0 or updates == total:
                checkpoint(latest)
                if config["smoke"] and updates == config["checkpoint_every_updates"]:
                    expected_hash = state_hash(model.state_dict())
                    expected_rng = json.dumps(rng.bit_generator.state, sort_keys=True)
                    restored = load_checkpoint(latest, model, optimizer, scaler, rng, config, branch_provenance)
                    if restored != updates or state_hash(model.state_dict()) != expected_hash or json.dumps(rng.bit_generator.state, sort_keys=True) != expected_rng:
                        raise ValueError("Execution checkpoint round-trip failed")
                    state["checkpoint_roundtrip_passed"] = True
                    report("checkpoint_roundtrip_passed")
        if branch != "M0":
            checkpoint(selected)
        if not selected.exists():
            raise ValueError("No selected checkpoint after completed budget")
        import resource
        report("complete", selected_checkpoint_sha256=sha256(selected),
               elapsed_this_invocation_seconds=time.monotonic() - started,
               process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    except BaseException as exc:
        # The last atomic checkpoint is authoritative. Do not serialize a partially failed update.
        report("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=type(exc).__name__ + ": " + str(exc))
        raise
    finally:
        if prefetch:
            reader.close()
        loader.clear()
        del model, optimizer, loss_fn, scaler
        torch.cuda.empty_cache()


def validate_execution_smoke(study, split):
    from .data import ROOT, sha256
    evidence = study["training"]["execution_evidence"]
    path = ROOT / evidence["run_path"]
    if sha256(path) != evidence["run_sha256"]:
        raise ValueError("Smoke evidence changed")
    run = json.loads(path.read_text())
    if run["status"] != "complete" or run["research_training"] or not run["provenance"]["smoke"]:
        raise ValueError("Execution smoke has not passed")
    if run["study_snapshot"]["labels"] != study["labels"] or run["provenance"]["split_sha256"] != sha256(ROOT / study["cohort"]["split_manifest"]):
        raise ValueError("Labels or split changed after smoke")
    for key in ("numeric_training_plan", "fixed_execution_plan", "paired_training_seeds"):
        if run["study_snapshot"]["training"][key] != study["training"][key]:
            raise ValueError("Training settings changed after smoke: " + key)
    starts = {}
    for branch, expected_hash in evidence["branch_status_sha256"].items():
        status_path = path.parent / branch / "status.json"
        if sha256(status_path) != expected_hash:
            raise ValueError("Branch smoke evidence changed")
        status = json.loads(status_path.read_text())
        state = status["trainer_state"]
        if status["status"] != "complete" or status["updates"] != status["budget"] or state["amp_skips"] or not state["checkpoint_roundtrip_passed"]:
            raise ValueError("Incomplete branch smoke")
        starts[branch] = state["initial_state_sha256"]
        validation_path = path.parent / branch / f"validation_{status['updates']:06d}.json"
        if sha256(validation_path) != evidence["validation_sha256"][branch]:
            raise ValueError("Smoke validation evidence changed")
        validation = json.loads(validation_path.read_text())
        for role in ("adult_validation", "pediatric_validation"):
            if set(validation[role]["cases"]) != set(split["partitions"][role]):
                raise ValueError("Smoke did not validate the full fixed cohort")
            values = np.asarray(list(validation[role]["cases"].values()))
            if values.shape != (len(split["partitions"][role]), len(study["labels"]) - 1) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                raise ValueError("Invalid validation Dice in smoke evidence")
    if set(starts) != {"M0", "M1", "M2"} or starts["M1"] != starts["M2"]:
        raise ValueError("Smoke branches did not use identical M0 initialization")


def save_source_snapshot(output, name="source_snapshot.tar.gz"):
    """실행 당시 코드를 복원할 수 있는 작은 archive. data와 generated artifact는 제외한다."""
    import tarfile
    from .data import ROOT, sha256
    paths = [ROOT / name for name in ("README.md", "pyproject.toml", "uv.lock", ".gitignore")]
    paths.extend(sorted((ROOT / "docs").glob("*.md")))
    for directory, pattern in (("src/tsv3", "*.py"), ("configs", "*.json"), ("tests", "*.py"), ("scripts", "*.py")):
        paths.extend(sorted((ROOT / directory).glob(pattern)))
    if name not in ("source_snapshot.tar.gz", "source_snapshot_fast.tar.gz", "source_snapshot_prefetch.tar.gz"):
        raise ValueError("Unexpected source snapshot name")
    destination = output / name
    if destination.exists():
        raise FileExistsError("Source snapshot already exists")
    temporary = destination.with_suffix(".gz.tmp")
    try:
        with tarfile.open(temporary, "w:gz") as archive:
            for path in paths:
                if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
                    raise ValueError("Source snapshot path escapes repository")
                archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256(destination)


def verify_loader_only_change(output, old_sources, new_sources):
    """Reject an in-place checkpoint migration if scientific functions or other modules changed."""
    import ast
    import tarfile
    from .data import sha256
    if set(new_sources) != set(old_sources) | {"fast_cache.py"}:
        raise ValueError("Unexpected source module added or removed")
    if any(new_sources[name] != digest for name, digest in old_sources.items() if name != "training.py"):
        raise ValueError("Non-loader source module changed")
    with tarfile.open(output / "source_snapshot.tar.gz") as archive:
        original_bytes = archive.extractfile("src/tsv3/training.py").read()
    if hashlib.sha256(original_bytes).hexdigest() != old_sources["training.py"]:
        raise ValueError("Original archived training source differs from provenance")
    original = ast.parse(original_bytes)
    current = ast.parse(Path(__file__).read_bytes())
    def protected(tree):
        return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and
                node.name not in {"run_branch", "save_source_snapshot", "main", "verify_loader_only_change",
                                  "verify_next_input_equivalence", "prepare_fast_cache_upgrade"}}
    if protected(original) != protected(current):
        raise ValueError("Scientific training function changed; checkpoint migration prohibited")


def verify_next_input_equivalence(checkpoint, old_cache, old_manifest, fast_cache, fast_manifest,
                                  normalization, cases, config, provenance):
    """Replay exactly the next sampled case, patch and augmentation from the old RNG state."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["format_version"] != 1 or payload["config"] != config or payload["provenance"] != {**provenance, "branch": "M0"}:
        raise ValueError("Original M0 checkpoint identity changed")
    if payload["trainer_state"] is None or payload["updates"] <= 0:
        raise ValueError("Original M0 checkpoint has no resumable training state")
    results = []
    for root, manifest in ((None, None), (fast_cache, fast_manifest)):
        rng = np.random.default_rng()
        rng.bit_generator.state = payload["numpy_rng"]
        torch.set_rng_state(payload["torch_rng"])
        if payload["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        case = cases[int(rng.integers(len(cases)))]
        loader = CaseCache(old_cache, old_manifest, normalization, [case], config["case_cache_gib"], root, manifest)
        data, seg, locations = loader.get(case)
        x, y = sample_patch(data, seg, config, rng, len(old_manifest["signature"]["labels"]), locations=locations)
        x, y = augment_patch(x, y, config["augmentation"], rng, len(old_manifest["signature"]["labels"]))
        results.append((case, x.detach().cpu(), y.detach().cpu(), json.dumps(rng.bit_generator.state, sort_keys=True)))
        loader.clear()
        del x, y, data, seg, locations
    one, two = results
    if one[0] != two[0] or not torch.equal(one[1], two[1]) or not torch.equal(one[2], two[2]) or one[3] != two[3]:
        raise ValueError("Old and fast cache produce different next training input")
    return {"checkpoint_update": payload["updates"], "next_case": one[0],
            "augmented_input_sha256": hashlib.sha256(one[1].numpy().tobytes()).hexdigest(),
            "target_sha256": hashlib.sha256(one[2].numpy().tobytes()).hexdigest(),
            "next_training_input_equal": True, "numpy_rng_equal": True}


def prepare_fast_cache_upgrade(output, previous, provenance, cache, manifest, normalization,
                               fast_cache, fast_manifest, study, plan):
    """One explicit, audited migration of an interrupted M0 checkpoint."""
    import shutil
    import tarfile
    from .data import sha256
    if previous["status"] != "interrupted" or previous["provenance"]["smoke"] or not previous["research_training"]:
        raise ValueError("Only interrupted research M0 may use the cache upgrade")
    if (output / "M0/selected.pt").exists() or (output / "M1").exists() or (output / "M2").exists():
        raise ValueError("Cache upgrade is restricted to M0 before model selection")
    old = previous["provenance"]
    for key, value in old.items():
        if key != "source_sha256" and provenance.get(key) != value:
            raise ValueError("Frozen study, data, environment or evaluation changed: " + key)
    if sha256(output / "source_snapshot.tar.gz") != previous["source_snapshot_sha256"]:
        raise ValueError("Original source snapshot changed")
    verify_loader_only_change(output, old["source_sha256"], provenance["source_sha256"])
    latest = output / "M0/latest.pt"
    if not latest.is_file():
        raise FileNotFoundError("Original M0 checkpoint missing")
    checkpoint_sha = sha256(latest)
    config = branch_config(study, "M0", False)
    equivalence = verify_next_input_equivalence(latest, cache, manifest, fast_cache, fast_manifest,
                                                 normalization, sorted(plan["split"]["partitions"]["M0_train"]), config, old)
    old_run = output / "run_before_fast_cache_upgrade.json"
    old_checkpoint = output / "M0/pre_fast_cache_upgrade.pt"
    if old_run.exists():
        if sha256(old_run) != sha256(output / "run.json"):
            raise ValueError("Original run backup differs")
    else:
        shutil.copy2(output / "run.json", old_run)
    if old_checkpoint.exists():
        if sha256(old_checkpoint) != checkpoint_sha:
            raise ValueError("Original checkpoint backup differs")
    else:
        shutil.copy2(latest, old_checkpoint)
    new_snapshot = output / "source_snapshot_fast.tar.gz"
    if new_snapshot.exists():
        with tarfile.open(new_snapshot) as archive:
            for name, digest in provenance["source_sha256"].items():
                if hashlib.sha256(archive.extractfile("src/tsv3/" + name).read()).hexdigest() != digest:
                    raise ValueError("Fast source snapshot differs from current code")
    else:
        save_source_snapshot(output, "source_snapshot_fast.tar.gz")
    return {"old_provenance": old, "checkpoint_sha256": checkpoint_sha,
            "original_run_sha256": sha256(old_run), "original_source_snapshot_sha256": previous["source_snapshot_sha256"],
            "fast_source_snapshot_sha256": sha256(new_snapshot),
            "fast_manifest_sha256": sha256(fast_cache / "manifest.json"), "next_input_equivalence": equivalence,
            "upgraded_utc": datetime.now(timezone.utc).isoformat()}


def verify_prefetch_source_change(output, old_sources, new_sources):
    import ast
    import tarfile
    from .data import sha256
    previous = json.loads((output / "run.json").read_text())
    archive_path = output / "source_snapshot.tar.gz"
    if sha256(archive_path) != previous["source_snapshot_sha256"]:
        raise ValueError("Original source archive changed")
    if set(old_sources) != set(new_sources) or any(
            old_sources[name] != new_sources[name] for name in old_sources if name not in {"training.py", "probe.py"}):
        raise ValueError("Only training orchestration and mechanical probe may change")
    with tarfile.open(archive_path) as archive:
        raw = archive.extractfile("src/tsv3/training.py").read()
    if hashlib.sha256(raw).hexdigest() != old_sources["training.py"]:
        raise ValueError("Archived training source does not match run provenance")
    allowed = {"CasePrefetcher", "training_input", "run_branch", "save_source_snapshot", "main", "execute_training",
               "verify_prefetch_source_change", "prepare_prefetch_upgrade"}
    def protected(tree):
        return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name not in allowed}
    if protected(ast.parse(raw)) != protected(ast.parse(Path(__file__).read_bytes())):
        raise ValueError("Scientific functions changed; prefetch migration prohibited")


def prepare_prefetch_upgrade(output, previous, provenance, evidence):
    import shutil
    from .data import sha256
    if (previous["status"] != "interrupted" or not previous["research_training"] or previous.get("cache_upgrade")
            or (output / "M0/selected.pt").exists() or (output / "M1").exists() or (output / "M2").exists()):
        raise ValueError("Prefetch migration requires an interrupted M0 before checkpoint selection/fine-tuning")
    old = previous["provenance"]
    if old.get("input_pipeline", "synchronous") != "synchronous" or provenance["input_pipeline"] != "case_prefetch_v1":
        raise ValueError("Unexpected input pipeline migration")
    for key, value in old.items():
        if key not in {"source_sha256", "input_pipeline"} and provenance.get(key) != value:
            raise ValueError("Frozen training/data/environment changed: " + key)
    verify_prefetch_source_change(output, old["source_sha256"], provenance["source_sha256"])
    report = json.loads(evidence.read_text())
    checkpoint = output / "M0/latest.pt"
    if (report["status"] != "throughput_benchmark_complete" or not report["prefetch_recommended"]
            or not report["input_equivalence"]["equal"] or not report["one_update_weights_close"]
            or report["source_sha256"] != provenance["source_sha256"] or report["original_provenance"] != old
            or report["checkpoint_sha256"] != sha256(checkpoint) or Path(report["run"]).resolve() != output.resolve()):
        raise ValueError("Matching equivalence and successful throughput benchmark are required")
    originals = [(output / "run.json", output / "run_before_prefetch_upgrade.json"),
                 (checkpoint, output / "M0/pre_prefetch_upgrade.pt"),
                 (evidence, output / "prefetch_benchmark.json")]
    for source, destination in originals:
        if destination.exists():
            if sha256(source) != sha256(destination):
                raise ValueError("Prefetch migration backup differs: " + str(destination))
        else:
            shutil.copy2(source, destination)
    snapshot = output / "source_snapshot_prefetch.tar.gz"
    if snapshot.exists():
        import tarfile
        with tarfile.open(snapshot) as archive:
            for name, digest in provenance["source_sha256"].items():
                if hashlib.sha256(archive.extractfile("src/tsv3/" + name).read()).hexdigest() != digest:
                    raise ValueError("Prefetch source snapshot differs")
    else:
        save_source_snapshot(output, snapshot.name)
    return {"old_provenance": old, "checkpoint_sha256": sha256(checkpoint),
            "original_run_sha256": sha256(output / "run_before_prefetch_upgrade.json"),
            "benchmark_sha256": sha256(output / "prefetch_benchmark.json"),
            "source_snapshot_sha256": sha256(snapshot), "upgraded_utc": datetime.now(timezone.utc).isoformat()}


def main():
    import argparse
    import contextlib
    import fcntl
    from .data import ROOT, DEFAULT_RAW
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "configs/study.json")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--review", type=Path, default=ROOT / "manifests/review_v300")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/preprocessed/abdominal_v1")
    parser.add_argument("--fast-cache", type=Path, help="Completed lossless Blosc cache for the same frozen cases")
    parser.add_argument("--upgrade-fast-cache", action="store_true", help="Audit and migrate only an interrupted M0 checkpoint")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, help="Declared training seed; the frozen cohort split stays unchanged")
    parser.add_argument("--prefetch", action="store_true", help="Prepare the next CT in one bounded CPU worker")
    parser.add_argument("--upgrade-prefetch", action="store_true", help="Migrate a paused M0 only after an equivalent, faster benchmark")
    parser.add_argument("--prefetch-evidence", type=Path, help="Throughput benchmark result.json for an explicit migration")
    parser.add_argument("--host-volume", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Separate 32-update branches; all development validation cases; no research weights")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    # Hold shared locks throughout training so maintenance cannot remove arrays in use.
    with contextlib.ExitStack() as locks:
        for root in dict.fromkeys(p for p in (args.cache, args.fast_cache) if p is not None):
            lock = locks.enter_context((root / ".lock").open("a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Cache conversion or maintenance is active: {root}") from exc
        execute_training(args, parser)


def execute_training(args, parser):
    import fcntl
    import platform
    import sys
    import tempfile
    from .data import ROOT, sha256, git_provenance
    from .preprocessing import load_plan, atomic_json
    from .night import verify_completion
    from .evaluation import validate_evaluation_contract
    plan = load_plan(args)
    manifest, normalization = verify_completion(args.cache, plan)
    if args.upgrade_fast_cache and (args.smoke or args.fast_cache is None or not args.resume):
        parser.error("--upgrade-fast-cache requires research --resume and --fast-cache")
    if args.upgrade_prefetch and (not args.prefetch or not args.resume or args.smoke or args.fast_cache is None or args.upgrade_fast_cache):
        parser.error("--upgrade-prefetch requires research --resume --prefetch --fast-cache and no cache upgrade")
    fast_manifest = None
    if args.fast_cache is not None:
        from .fast_cache import verify_fast_cache
        fast_manifest = verify_fast_cache(args.fast_cache, args.cache, manifest)
    elif any(not (args.cache / record["file"]).is_file() for record in manifest["completed"].values()):
        raise ValueError("Original cache arrays are missing; specify the verified --fast-cache or regenerate the original arrays")
    study = plan["study"]
    seed = training_seed(study, args.seed)
    if study["training"]["numeric_training_plan"] is None:
        raise ValueError("Numeric training plan must be fixed first")
    if not args.smoke and not args.dry_run and study["status"] != "ready_for_research_training":
        raise ValueError("Research training gated: first pass execution smoke and freeze evaluation protocol")
    evaluation_identity = None
    if not args.smoke and study["status"] == "ready_for_research_training":
        validate_execution_smoke(study, plan["split"])
        evaluation_identity = validate_evaluation_contract(study)
    output = args.output or training_output(study, seed, args.smoke, args.seed is not None)
    check_output_seed(output, seed)
    if args.dry_run:
        print(json.dumps({"status": "plan_only", "output": str(output), "cases": len(manifest["completed"]),
                          "seed": seed, "split_sha256": plan["signature"]["split_sha256"],
                          "normalization_sha256": manifest["normalization"]["sha256"],
                          "smoke": args.smoke, "branch_budgets": {b: branch_config(study, b, args.smoke, seed)["optimizer_updates"] for b in ("M0", "M1", "M2")},
                          "production_training_enabled": study["status"] == "ready_for_research_training"}, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if output.exists() and any(p.name != ".lock" for p in output.iterdir()) and not args.resume:
        raise FileExistsError("Output exists; use --resume or a new output path")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(study["training"]["numeric_training_plan"]["cpu_threads"])
    if args.fast_cache is not None:
        import blosc2
        blosc2.set_nthreads(study["training"]["numeric_training_plan"]["cpu_threads"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    provenance = {"study_sha256": sha256(args.study), "manifest_sha256": sha256(args.cache / "manifest.json"),
                  "normalization_sha256": manifest["normalization"]["sha256"], "split_sha256": plan["signature"]["split_sha256"],
                  "source_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                  "lock_sha256": sha256(ROOT / "uv.lock"), "torch": torch.__version__, "cuda": torch.version.cuda,
                  "gpu": torch.cuda.get_device_name(), "smoke": args.smoke, "training_seed": seed,
                  "input_pipeline": "case_prefetch_v1" if args.prefetch else "synchronous",
                  "evaluation_identity": evaluation_identity}
    if args.fast_cache is not None:
        provenance["fast_manifest_sha256"] = sha256(args.fast_cache / "manifest.json")
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_path = output / "run.json"
        source_snapshot_sha256 = None
        cache_upgrade = None
        prefetch_upgrade = None
        if run_path.exists():
            previous = json.loads(run_path.read_text())
            if previous["provenance"] != provenance:
                if args.upgrade_prefetch:
                    evidence = args.prefetch_evidence or ROOT / "outputs" / output.name.replace("training_", "throughput_", 1) / "result.json"
                    prefetch_upgrade = prepare_prefetch_upgrade(output, previous, provenance, evidence)
                elif not args.upgrade_fast_cache or args.fast_cache is None:
                    raise ValueError("Source, configuration or data changed since this run")
                else:
                    cache_upgrade = prepare_fast_cache_upgrade(output, previous, provenance, args.cache, manifest,
                                                               normalization, args.fast_cache, fast_manifest, study, plan)
            else:
                cache_upgrade = previous.get("cache_upgrade")
                prefetch_upgrade = previous.get("prefetch_upgrade")
                if args.upgrade_fast_cache and cache_upgrade is None:
                    raise ValueError("No interrupted original run to upgrade")
                if args.upgrade_prefetch and prefetch_upgrade is None:
                    raise ValueError("No interrupted original run to upgrade")
            if previous["status"] == "complete":
                print(json.dumps({"status": "already_complete", "output": str(output)}))
                return
            source_snapshot_sha256 = previous.get("source_snapshot_sha256")
            if not args.smoke and (not source_snapshot_sha256 or sha256(output / "source_snapshot.tar.gz") != source_snapshot_sha256):
                raise ValueError("Archived source snapshot changed or is missing")
            if cache_upgrade:
                if (sha256(output / "M0/pre_fast_cache_upgrade.pt") != cache_upgrade["checkpoint_sha256"]
                        or sha256(output / "run_before_fast_cache_upgrade.json") != cache_upgrade["original_run_sha256"]
                        or sha256(output / "source_snapshot_fast.tar.gz") != cache_upgrade["fast_source_snapshot_sha256"]
                        or provenance["fast_manifest_sha256"] != cache_upgrade["fast_manifest_sha256"]):
                    raise ValueError("Fast cache upgrade audit artifact changed")
            if prefetch_upgrade:
                for path, key in (("M0/pre_prefetch_upgrade.pt", "checkpoint_sha256"),
                                  ("run_before_prefetch_upgrade.json", "original_run_sha256"),
                                  ("prefetch_benchmark.json", "benchmark_sha256"),
                                  ("source_snapshot_prefetch.tar.gz", "source_snapshot_sha256")):
                    if sha256(output / path) != prefetch_upgrade[key]:
                        raise ValueError("Prefetch migration audit artifact changed: " + path)
        elif not args.smoke:
            source_snapshot_sha256 = save_source_snapshot(output)
        record = {"status": "running", "provenance": provenance, "git": git_provenance(),
                  "study_snapshot": study, "research_training": not args.smoke,
                  "source_snapshot_sha256": source_snapshot_sha256,
                  "cache_upgrade": cache_upgrade,
                  "prefetch_upgrade": prefetch_upgrade,
                  "started_this_invocation_utc": datetime.now(timezone.utc).isoformat(),
                  "environment": {"python": sys.version, "platform": platform.platform(),
                                  "visible_cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads()}}
        atomic_json(run_path, record)
        # Smoke weights are temporary; completed JSON/log evidence stays in the stable output directory.
        temporary = tempfile.TemporaryDirectory(prefix=".smoke-weights-", dir=output) if args.smoke else None
        work = Path(temporary.name) if temporary else output
        try:
            for branch in ("M0", "M1", "M2"):
                migration = prefetch_upgrade or cache_upgrade
                run_branch(branch, plan, args.cache, work, branch_config(study, branch, args.smoke, seed), provenance,
                           work / "M0/selected.pt", args.resume and not args.smoke, args.host_volume,
                           args.fast_cache, fast_manifest,
                           {"provenance": migration["old_provenance"], "checkpoint_sha256": migration["checkpoint_sha256"]}
                           if migration else None, prefetch=args.prefetch)
            record["status"] = "complete"
        except BaseException as exc:
            record.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=type(exc).__name__ + ": " + str(exc))
            raise
        finally:
            if temporary:
                # Preserve failure logs as well, but no throwaway model files.
                import shutil
                for path in work.rglob("*.json"):
                    destination = output / path.relative_to(work)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, destination)
                if (work / "run.log").exists():
                    shutil.copy2(work / "run.log", output / "run.log")
                temporary.cleanup()
            record["finished_this_invocation_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_json(run_path, record)
        print(json.dumps({"status": "execution_smoke_passed_not_research_training" if args.smoke else "training_complete",
                          "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
