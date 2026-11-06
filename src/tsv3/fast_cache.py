"""Lossless, resumable Blosc copy of the frozen development NPZ cache."""

import argparse
import fcntl
import json
from pathlib import Path

import blosc2
import numpy as np

from .data import ROOT, sha256
from .preprocessing import atomic_json, check_space


FORMAT = "blosc2_zstd1_shuffle_npz_stored_v1"
KEYS = ("data", "seg", "foreground_locations")


def read_case(path):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {key + "_blob" for key in KEYS}:
            raise ValueError("Unexpected fast cache fields: " + str(path))
        return tuple(blosc2.unpack_array(archive[key + "_blob"]) for key in KEYS)


def convert_case(source, destination):
    with np.load(source, allow_pickle=False) as archive:
        originals = tuple(archive[key] for key in KEYS)
    blobs = {key + "_blob": np.frombuffer(blosc2.pack_array(array, clevel=1,
             filter=blosc2.Filter.SHUFFLE, codec=blosc2.Codec.ZSTD), dtype=np.uint8)
             for key, array in zip(KEYS, originals)}
    temporary = destination.with_suffix(".npz.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **blobs)
        restored = read_case(temporary)
        if any(a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b)
               for a, b in zip(originals, restored)):
            raise ValueError("Lossless cache round-trip failed: " + source.name)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"file": destination.name, "bytes": destination.stat().st_size,
            "sha256": sha256(destination), "source_sha256": sha256(source)}


def verify_fast_cache(root, source, source_manifest):
    path = root / "manifest.json"
    fast = json.loads(path.read_text())
    if fast["status"] != "complete" or fast["format"] != FORMAT or fast["error"] is not None:
        raise ValueError("Fast cache is incomplete")
    if fast["blosc2_version"] != blosc2.__version__:
        raise ValueError("Fast cache compressor version changed")
    if fast["source_manifest_sha256"] != sha256(source / "manifest.json"):
        raise ValueError("Fast cache source manifest changed")
    if fast["normalization_sha256"] != source_manifest["normalization"]["sha256"]:
        raise ValueError("Fast cache normalization source changed")
    if set(fast["completed"]) != set(source_manifest["completed"]):
        raise ValueError("Fast cache case set differs from development cache")
    for case, record in fast["completed"].items():
        original = source_manifest["completed"][case]
        item = root / (case + ".npz")
        if (record["file"] != item.name or record["source_sha256"] != original["sha256"]
                or not item.is_file() or item.stat().st_size != record["bytes"]):
            raise ValueError("Fast cache file or source differs: " + case)
    return fast


def validate_arrays(arrays, shape, label_count):
    data, seg, locations = arrays
    expected_shape = (1, *shape)
    if (data.dtype != np.float32 or seg.dtype != np.uint8 or data.shape != expected_shape
            or seg.shape != expected_shape or not np.isfinite(data).all()):
        raise ValueError("Invalid cached data/target shape, dtype or finite values")
    present = set(np.unique(seg).tolist())
    labels = set(range(1, label_count + 1))
    if not labels <= present or not present <= labels | {0, label_count + 1}:
        raise ValueError("Cached target labels differ from the fixed panel")
    if (locations.dtype != np.int32 or locations.ndim != 2 or locations.shape[1] != 4
            or not len(locations) or set(locations[:, 0].tolist()) != labels):
        raise ValueError("Invalid cached foreground locations")
    xyz = locations[:, 1:]
    if np.any(xyz < 0) or np.any(xyz >= np.asarray(shape)):
        raise ValueError("Foreground coordinates outside the target grid")
    if not np.array_equal(seg[(np.zeros(len(xyz), dtype=int), *xyz.T)], locations[:, 0]):
        raise ValueError("Foreground coordinates do not match target labels")


def verify_and_prune(source, output, prune_source=False):
    """Verify every fast array before removing any explicitly listed source NPZ."""
    import contextlib
    import fcntl
    from datetime import datetime, timezone

    if source.is_symlink() or output.is_symlink():
        raise ValueError("Maintenance requires real cache directories, not symlinks")
    source, output = source.resolve(), output.resolve()
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Cache directories must be separate")
    blosc2.set_nthreads(4)
    with contextlib.ExitStack() as locks:
        for root in (source, output):
            lock = locks.enter_context((root / ".lock").open("a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Cache is in use; stop training/conversion before maintenance: {root}") from exc
        original = json.loads((source / "manifest.json").read_text())
        if original["status"] != "development_cache_complete_not_training_results" or original["error"] is not None:
            raise ValueError("Original cache manifest is not complete")
        if (original["normalization"]["file"] != "normalization.json"
                or sha256(source / "normalization.json") != original["normalization"]["sha256"]):
            raise ValueError("Normalization file changed")
        fast = verify_fast_cache(output, source, original)
        fast_hash, original_hash = sha256(output / "manifest.json"), sha256(source / "manifest.json")
        label_count = len(original["signature"]["labels"]) - 1
        report_path = output / "verification.json"
        report = {"status": "verifying", "started_utc": datetime.now(timezone.utc).isoformat(),
                  "source_manifest_sha256": original_hash, "fast_manifest_sha256": fast_hash,
                  "normalization_sha256": original["normalization"]["sha256"],
                  "total": len(fast["completed"]), "verified": 0, "prune_source_requested": prune_source,
                  "removed_source_cases": [], "removed_source_bytes": 0, "source_arrays_already_absent": 0,
                  "error": None}
        atomic_json(report_path, report)
        removable = []
        case = None
        try:
            for case, record in sorted(fast["completed"].items()):
                original_record = original["completed"][case]
                if (Path(case).name != case or record["file"] != case + ".npz"
                        or original_record["file"] != case + ".npz"):
                    raise ValueError("Unexpected case path in cache manifest")
                path, original_path = output / record["file"], source / original_record["file"]
                if path.is_symlink() or original_path.is_symlink():
                    raise ValueError("Cache array symlinks are not allowed")
                if sha256(path) != record["sha256"]:
                    raise ValueError("Fast array hash mismatch")
                arrays = read_case(path)
                validate_arrays(arrays, original_record["properties"]["preprocessed_shape"], label_count)
                del arrays
                if original_path.exists():
                    stat = original_path.stat()
                    if stat.st_size != original_record["bytes"] or sha256(original_path) != original_record["sha256"]:
                        raise ValueError("Source array hash/size mismatch")
                    removable.append((case, original_path, stat))
                else:
                    report["source_arrays_already_absent"] += 1
                report["verified"] += 1
                if report["verified"] % 25 == 0 or report["verified"] == report["total"]:
                    atomic_json(report_path, report)
                    print(json.dumps({"verified": case, "done": report["verified"], "total": report["total"]}), flush=True)
            if (sha256(output / "manifest.json") != fast_hash or sha256(source / "manifest.json") != original_hash
                    or sha256(source / "normalization.json") != report["normalization_sha256"]):
                raise ValueError("Cache metadata changed during verification")
            report.update(status="verified", source_arrays_present=len(removable),
                          source_bytes_available_to_prune=sum(s.st_size for _, _, s in removable))
            atomic_json(report_path, report)
            if prune_source:
                report["status"] = "pruning_source"
                atomic_json(report_path, report)
                for case, path, before in removable:
                    after = path.stat()
                    if path.is_symlink() or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                        raise ValueError("Source array changed after verification")
                    path.unlink()
                    report["removed_source_cases"].append(case)
                    report["removed_source_bytes"] += before.st_size
                    atomic_json(report_path, report)
                report["status"] = "verified_source_arrays_pruned"
            report["finished_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_json(report_path, report)
        except BaseException as exc:
            report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                          error=f"{case}: {type(exc).__name__}: {exc}")
            atomic_json(report_path, report)
            raise
    return report


def run_conversion(source, output, host_volume=None, limit=None):
    source, output = source.resolve(), output.resolve()
    if source == output or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Fast cache must be separate from the original cache")
    original = json.loads((source / "manifest.json").read_text())
    if original["status"] != "development_cache_complete_not_training_results" or original["error"] is not None:
        raise ValueError("Original cache is not complete")
    output.mkdir(parents=True, exist_ok=True)
    blosc2.set_nthreads(4)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            fast = json.loads(manifest_path.read_text())
            if fast["format"] != FORMAT or fast["source_manifest_sha256"] != sha256(source / "manifest.json"):
                raise ValueError("Fast cache contract changed")
        else:
            fast = {"format": FORMAT, "status": "running", "error": None,
                    "source_manifest_sha256": sha256(source / "manifest.json"),
                    "normalization_sha256": original["normalization"]["sha256"],
                    "blosc2_version": blosc2.__version__, "completed": {}}
            atomic_json(manifest_path, fast)
        if fast["blosc2_version"] != blosc2.__version__ or fast["normalization_sha256"] != original["normalization"]["sha256"]:
            raise ValueError("Fast cache compressor or normalization changed")
        if not set(fast["completed"]) <= set(original["completed"]):
            raise ValueError("Unexpected case in fast cache")
        try:
            fast["error"] = None
            for case, record in fast["completed"].items():
                destination = output / (case + ".npz")
                if (record["file"] != destination.name or not destination.is_file()
                        or destination.stat().st_size != record["bytes"] or sha256(destination) != record["sha256"]
                        or record["source_sha256"] != original["completed"][case]["sha256"]):
                    raise ValueError("Completed fast cache changed: " + case)
            pending = sorted(set(original["completed"]) - set(fast["completed"]))
            if limit is not None:
                pending = pending[:limit]
            for case in pending:
                source_file = source / (case + ".npz")
                record = original["completed"][case]
                if (record["file"] != source_file.name or source_file.stat().st_size != record["bytes"]
                        or sha256(source_file) != record["sha256"]):
                    raise ValueError("Original cache changed: " + case)
                check_space(output, host_volume, max(512 * 2**20, source_file.stat().st_size * 2), 10)
                result = convert_case(source_file, output / (case + ".npz"))
                fast["completed"][case] = result
                fast.update(status="running", error=None)
                atomic_json(manifest_path, fast)
                print(json.dumps({"converted": case, "done": len(fast["completed"]),
                                  "total": len(original["completed"])}), flush=True)
            fast["status"] = "complete" if len(fast["completed"]) == len(original["completed"]) else "partial"
            atomic_json(manifest_path, fast)
        except BaseException as exc:
            fast.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error=type(exc).__name__ + ": " + str(exc))
            atomic_json(manifest_path, fast)
            raise
    return fast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/preprocessed/abdominal_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "data/preprocessed/abdominal_v1_fast")
    parser.add_argument("--host-volume", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--verify", action="store_true", help="Hash and decompress every fast array; retain metadata")
    parser.add_argument("--prune-source", action="store_true", help="With --verify, remove original NPZ arrays only after all cases pass")
    args = parser.parse_args()
    if args.prune_source and not args.verify:
        parser.error("--prune-source requires --verify")
    if args.verify:
        if args.resume or args.limit is not None:
            parser.error("--verify cannot be combined with conversion --resume or --limit")
        report = verify_and_prune(args.source, args.output, args.prune_source)
        print(json.dumps({"status": report["status"], "verified": report["verified"],
                          "removed_source_GiB": report["removed_source_bytes"] / 2**30,
                          "report": str(args.output / "verification.json")}), flush=True)
        return
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.output.exists() and not args.resume:
        raise FileExistsError("Fast cache exists; use --resume")
    result = run_conversion(args.source, args.output, args.host_volume, args.limit)
    print(json.dumps({"status": result["status"], "cases": len(result["completed"])}), flush=True)


if __name__ == "__main__":
    main()
