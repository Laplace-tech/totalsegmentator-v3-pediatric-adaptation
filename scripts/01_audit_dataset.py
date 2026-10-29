
"""
TotalSegmentator V3 - Dataset Audit (Stage 1)

Purpose:
    Inspect dataset structure, case inventory,
    segmentation label coverage, and metadata candidates.

Properties:
    - Read-only access to raw dataset
    - No third-party dependencies
    - Deterministic inventory generation
    - No loading of medical image voxel arrays

Usage:
    python3 scripts/01_audit_dataset.py
"""

import argparse
import csv
import json
import os
import re

from pathlib import Path
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Totalsegmentator_dataset_v300"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "manifests"
    / "audit_v300"
)

METADATA_EXTENSIONS = {
    ".json", ".jsonl", ".csv", ".tsv",
    ".txt", ".xlsx", ".xls",
    ".yaml", ".yml", ".parquet",
}


def file_extension(filename):
    name = filename.lower()

    if name.endswith(".nii.gz"):
        return ".nii.gz"

    return Path(filename).suffix.lower() or "(none)"


def save_csv(path, rows, columns):
    with path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns
        )
        writer.writeheader()
        writer.writerows(rows)


def audit(data_root):
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"Dataset not found: {data_root}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    cases = sorted(
        (
            path for path in data_root.iterdir()
            if path.is_dir()
        ),
        key=lambda path: path.name
    )

    inventory = []
    label_counts = Counter()
    extension_counts = Counter()
    metadata_candidates = []

    total_nifti = 0
    total_masks = 0
    total_zero_bytes = 0

    missing_ct = []
    missing_segmentations = []
    zero_byte_examples = []

    # Metadata files directly under dataset root
    root_files = sorted(
        (
            path for path in data_root.iterdir()
            if path.is_file()
        ),
        key=lambda path: path.name
    )

    for path in root_files:
        ext = file_extension(path.name)

        if ext in METADATA_EXTENSIONS:
            metadata_candidates.append({
                "case_id": "",
                "relative_path": path.name,
                "extension": ext
            })

    print("=" * 55)
    print("TotalSegmentator V3 - DATA AUDIT")
    print("=" * 55)
    print(f"Detected directories: {len(cases)}")

    for index, case in enumerate(cases, start=1):

        labels = set()
        nonseg_nifti = []
        case_metadata = []

        n_nifti = 0
        n_masks = 0
        n_zero_bytes = 0

        for current, directories, filenames in os.walk(
            case,
            followlinks=False
        ):
            directories.sort()

            for filename in sorted(filenames):
                path = Path(current) / filename
                relative = path.relative_to(case).as_posix()

                ext = file_extension(filename)
                extension_counts[ext] += 1

                if path.stat().st_size == 0:
                    n_zero_bytes += 1
                    total_zero_bytes += 1

                    if len(zero_byte_examples) < 20:
                        zero_byte_examples.append(
                            f"{case.name}/{relative}"
                        )

                if ext in {".nii.gz", ".nii"}:
                    n_nifti += 1
                    total_nifti += 1

                    if path.parent.name == "segmentations":
                        if ext == ".nii.gz":
                            label = filename[:-7]
                        else:
                            label = filename[:-4]

                        labels.add(label)
                        n_masks += 1
                        total_masks += 1
                    else:
                        nonseg_nifti.append(relative)

                elif ext in METADATA_EXTENSIONS:
                    case_metadata.append(relative)

                    metadata_candidates.append({
                        "case_id": case.name,
                        "relative_path": (
                            f"{case.name}/{relative}"
                        ),
                        "extension": ext
                    })

        label_counts.update(labels)

        has_ct = (case / "ct.nii.gz").is_file()

        if not has_ct:
            missing_ct.append(case.name)

        if not (case / "segmentations").is_dir():
            missing_segmentations.append(case.name)

        inventory.append({
            "case_id": case.name,
            "has_ct_nifti": int(has_ct),
            "n_nifti": n_nifti,
            "n_mask_files": n_masks,
            "n_unique_labels": len(labels),
            "n_nonseg_nifti": len(nonseg_nifti),
            "nonseg_nifti_files": ";".join(
                sorted(nonseg_nifti)
            ),
            "metadata_files": ";".join(
                sorted(case_metadata)
            ),
            "n_zero_byte_files": n_zero_bytes
        })

        if index % 100 == 0 or index == len(cases):
            print(
                f"Scanned {index:,}/{len(cases):,} cases"
            )

    # Case-level manifest
    save_csv(
        OUTPUT_DIR / "case_inventory.csv",
        inventory,
        [
            "case_id",
            "has_ct_nifti",
            "n_nifti",
            "n_mask_files",
            "n_unique_labels",
            "n_nonseg_nifti",
            "nonseg_nifti_files",
            "metadata_files",
            "n_zero_byte_files"
        ]
    )

    # Segmentation label coverage
    coverage = [
        {
            "label": label,
            "n_cases": count,
            "coverage_percent": round(
                count / len(cases) * 100, 2
            ) if cases else 0
        }
        for label, count in sorted(
            label_counts.items(),
            key=lambda item: (-item[1], item[0])
        )
    ]

    save_csv(
        OUTPUT_DIR / "label_coverage.csv",
        coverage,
        [
            "label",
            "n_cases",
            "coverage_percent"
        ]
    )

    # Metadata candidates
    save_csv(
        OUTPUT_DIR / "metadata_candidates.csv",
        metadata_candidates,
        [
            "case_id",
            "relative_path",
            "extension"
        ]
    )

    # File type statistics
    save_csv(
        OUTPUT_DIR / "file_types.csv",
        [
            {
                "extension": ext,
                "count": count
            }
            for ext, count in sorted(
                extension_counts.items()
            )
        ],
        ["extension", "count"]
    )

    unexpected_directories = [
        case.name for case in cases
        if re.fullmatch(r"s\d+", case.name) is None
    ]

    summary = {
        "dataset": "Totalsegmentator_dataset_v300",
        "audit_stage": "file_inventory",
        "total_case_directories": len(cases),
        "total_nifti_files": total_nifti,
        "total_segmentation_files": total_masks,
        "unique_label_names": len(label_counts),
        "metadata_candidate_files": len(
            metadata_candidates
        ),
        "zero_byte_files": total_zero_bytes,
        "cases_without_ct_nifti": missing_ct,
        "cases_without_segmentations_dir": (
            missing_segmentations
        ),
        "unexpected_directory_names": (
            unexpected_directories
        ),
        "zero_byte_examples": zero_byte_examples,
        "root_files": [
            path.name for path in root_files
        ],
        "limitations": [
            "NIfTI voxel contents were not inspected.",
            "Mask non-emptiness was not verified.",
            "Adult/pediatric cohorts were not identified.",
            "Compressed file integrity was not verified."
        ]
    }

    with (
        OUTPUT_DIR / "audit_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True
        )

    print()
    print("=" * 55)
    print("AUDIT SUMMARY")
    print("=" * 55)
    print(f"Cases:            {len(cases):,}")
    print(f"NIfTI files:      {total_nifti:,}")
    print(f"Segmentation files: {total_masks:,}")
    print(f"Unique labels:    {len(label_counts):,}")
    print(f"Metadata files:   {len(metadata_candidates):,}")
    print(f"Zero-byte files:  {total_zero_bytes:,}")
    print(f"Missing ct.nii.gz: {len(missing_ct):,}")
    print(
        "Missing segmentation directories: "
        f"{len(missing_segmentations):,}"
    )
    print()
    print("TOP 15 LABELS")

    for label, count in label_counts.most_common(15):
        print(f"  {label:<32} {count:,}")

    print()
    print(f"Results saved: {OUTPUT_DIR}")
    print("Stage 1 audit completed.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="TotalSegmentator V3 Dataset Audit"
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Path to extracted TotalSegmentator dataset"
    )

    args = parser.parse_args()
    audit(args.data.expanduser().resolve())
