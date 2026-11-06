"""Manifest 입력과 연구 설정을 읽는 공통 구현. Raw 디렉토리를 재열거하지 않는다."""

import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = ROOT / "data/raw/Totalsegmentator_dataset_v300"
DEFAULT_INVENTORY = ROOT / "manifests/audit_v300"
DEFAULT_PROTOCOL = ROOT / "configs/protocol.json"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_csv(path, delimiter=","):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def write_csv(path, rows, columns):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_inputs(raw=DEFAULT_RAW, inventory=DEFAULT_INVENTORY, protocol=DEFAULT_PROTOCOL):
    raw, inventory, protocol = Path(raw), Path(inventory), Path(protocol)
    cases = read_csv(inventory / "case_inventory.csv")
    metadata = read_csv(raw / "meta.csv", delimiter=";")
    case_ids = [row["case_id"] for row in cases]
    meta_ids = [row["image_id"] for row in metadata]
    if len(set(case_ids)) != len(case_ids) or len(set(meta_ids)) != len(meta_ids):
        raise ValueError("Duplicate case IDs in inventory or metadata")
    if set(case_ids) != set(meta_ids):
        raise ValueError("Inventory and metadata case sets differ")
    if any(Path(x).name != x or x in {".", ".."} for x in case_ids):
        raise ValueError("Unsafe case ID")
    labels = sorted(row["label"] for row in read_csv(inventory / "label_coverage.csv"))
    if len(labels) != len(set(labels)) or not labels:
        raise ValueError("Invalid label inventory")
    settings = json.loads(protocol.read_text(encoding="utf-8"))
    provenance = {
        "metadata_sha256": sha256(raw / "meta.csv"),
        "inventory_sha256": sha256(inventory / "case_inventory.csv"),
        "labels_sha256": sha256(inventory / "label_coverage.csv"),
        "protocol_sha256": sha256(protocol),
    }
    return case_ids, {row["image_id"]: row for row in metadata}, labels, settings, provenance


def classify_metadata(row, settings):
    """Age 기반 작업 정의. 환자 독립성이나 공식 291건의 소속을 보장하지 않는다."""
    value = row["age"].strip()
    if not value:
        return "unknown", ["missing_age"]
    try:
        age = float(value)
    except ValueError:
        return "unknown", ["invalid_age"]
    if not math.isfinite(age) or age < 0:
        return "unknown", ["invalid_age"]
    cohort = "pediatric" if age < settings["pediatric_age_upper_exclusive"] else "adult"
    reasons = ["age_requires_source_review"] if age > settings["age_review_above"] else []
    return cohort, reasons


def git_provenance():
    def run(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    return {"git_commit": run("rev-parse", "HEAD"), "git_dirty": bool(run("status", "--porcelain"))}
