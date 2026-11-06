"""실험 누수 방지와 NIfTI geometry 계약을 검사한다."""

import unittest
import tempfile
import copy
import json
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import nibabel as nib
import numpy as np

from tsv3.audit import geometry, audit_case
from tsv3.data import ROOT, classify_metadata
from tsv3.splits import allocation_counts, assess_cohorts, make_splits, validate_splits, study_eligibility
from tsv3.review import panel_labels, screen_cases, review_configuration
from tsv3.preprocessing import preprocess_case, run_cache, check_space


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.settings = {"seed": 42, "finetuning_cases_per_arm": 2,
                         "validation_cases_per_cohort": 1, "manual_training_exclusions": {"quarantine": "duplicate"},
                         "pediatric_age_upper_exclusive": 18, "age_review_above": 100}
        self.rows = []
        for cohort in ("adult", "pediatric"):
            for i in range(7):
                case_id = cohort + str(i)
                self.rows.append({"case_id": case_id, "cohort": cohort, "official_split": "test" if i == 6 else "train",
                                  "automated_eligible": "1", "ct_voxel_sha256": case_id})

    def test_m0_m2_disjoint_and_test_preserved(self):
        pools = make_splits(self.rows, self.settings)
        self.assertFalse(set(pools["M0_train"]) & set(pools["M2_finetune"]))
        self.assertEqual(pools["pediatric_test"], ["pediatric6"])
        self.assertEqual(len(pools["M1_finetune"]), len(pools["M2_finetune"]))
        self.assertEqual(pools, make_splits(self.rows, self.settings))

    def test_duplicate_ct_rejected_even_with_different_case_ids(self):
        self.rows[0]["ct_voxel_sha256"] = self.rows[6]["ct_voxel_sha256"]
        with self.assertRaisesRegex(ValueError, "Repeated CT"):
            make_splits(self.rows, self.settings)

    def test_study_screen_preserves_test_and_manual_statuses(self):
        rows = [dict(r, exclusion_reasons="") for r in self.rows]
        cases = {r["case_id"]: {"nonempty_no_border_organs": "1"} for r in rows if r["official_split"] == "train"}
        reviews = {case: [{"annotation": "pending", "coverage": "pending"}] for case in cases}
        reviews["adult0"][0]["annotation"] = "reject"
        reviews["pediatric0"][0]["coverage"] = "partial"
        cases["adult1"]["nonempty_no_border_organs"] = "0"
        originals = copy.deepcopy((rows, cases, reviews))
        screened = study_eligibility(rows, cases, reviews, 1)
        held = {r["case_id"] for r in screened if r["automated_eligible"] == "0"}
        self.assertEqual(held, {"adult0", "pediatric0", "adult1"})
        self.assertEqual((rows, cases, reviews), originals)
        self.assertEqual([r for r in screened if r["official_split"] == "test"],
                         [r for r in rows if r["official_split"] == "test"])

    def test_validator_rejects_silent_test_removal_and_validation_shrinkage(self):
        pools = make_splits(self.rows, self.settings)
        pools["excluded"].append(pools["adult_test"].pop())
        with self.assertRaisesRegex(ValueError, "silently excluded"):
            validate_splits(pools, self.rows, self.settings)
        pools = make_splits(self.rows, self.settings)
        pools["M0_train"].append(pools["adult_validation"].pop())
        with self.assertRaisesRegex(ValueError, "validation case budget"):
            validate_splits(pools, self.rows, self.settings)

    def test_age_boundary_and_unknown_not_inferred(self):
        self.assertEqual(classify_metadata({"age": "17"}, self.settings), ("pediatric", []))
        self.assertEqual(classify_metadata({"age": "18"}, self.settings), ("adult", []))
        self.assertEqual(classify_metadata({"age": ""}, self.settings)[0], "unknown")
        self.assertTrue(classify_metadata({"age": "118"}, self.settings)[1])

    def test_non_3d_image_rejected(self):
        image = nib.Nifti1Image(np.zeros((3, 3, 3, 2), dtype=np.uint8), np.eye(4))
        with self.assertRaisesRegex(ValueError, "not_nonempty_3d"):
            geometry(image)

    def test_empty_mask_is_recorded_without_excluding_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "example"
            (case / "segmentations").mkdir(parents=True)
            ct = np.arange(64, dtype=np.int16).reshape(4, 4, 4)
            nib.save(nib.Nifti1Image(ct, np.eye(4)), case / "ct.nii.gz")
            nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.uint8), np.eye(4)),
                     case / "segmentations/organ.nii.gz")
            result, organs = audit_case((tmp, "example", ["organ"], 0.1))
            self.assertEqual(result["technical_errors"], [])
            self.assertEqual(organs[0]["empty"], 1)
            self.assertIsNone(organs[0]["volume_mm3"])

    def test_actual_geometry_shift_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "example"
            (case / "segmentations").mkdir(parents=True)
            nib.save(nib.Nifti1Image(np.arange(64, dtype=np.int16).reshape(4, 4, 4), np.eye(4)), case / "ct.nii.gz")
            affine = np.eye(4)
            affine[0, 3] = 2.0
            nib.save(nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.uint8), affine), case / "segmentations/organ.nii.gz")
            result, organs = audit_case((tmp, "example", ["organ"], 0.1))
            self.assertTrue(result["technical_errors"])
            self.assertIn("geometry_mismatch", organs[0]["error"])

    def test_review_queue_excludes_test_and_does_not_auto_approve(self):
        cases = []
        organs = []
        for case_id, split, eligible, empty in (("train", "train", "1", "0"),
                                               ("test", "test", "1", "0"),
                                               ("held", "train", "0", "0"),
                                               ("empty", "train", "1", "1")):
            cases.append({"case_id": case_id, "official_split": split, "cohort": "pediatric",
                          "automated_eligible": eligible, "exclusion_reasons": "hold" if eligible == "0" else "",
                          "shape": "[4,4,4]"})
            organs.append({"case_id": case_id, "label": "liver", "empty": empty,
                           "touches_border": "0", "error": ""})
        screened, reviews, _ = screen_cases(cases, organs, ["liver"])
        self.assertEqual([r["case_id"] for r in reviews], ["train"])
        self.assertEqual(reviews[0]["coverage"], "pending")
        self.assertEqual(reviews[0]["annotation"], "pending")
        self.assertEqual(next(r for r in screened if r["case_id"] == "empty")["review_candidate"], 0)
        with self.assertRaisesRegex(ValueError, "Repeated"):
            screen_cases(cases, organs + [organs[0]], ["liver"])
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            screen_cases(cases, organs[:-1], ["liver"])

    def test_review_label_ids_cannot_collide(self):
        with self.assertRaisesRegex(ValueError, "Class IDs"):
            panel_labels({"labels": {"background": 0, "liver": 1, "spleen": 1}})

    def test_training_seed_change_does_not_change_review_inputs(self):
        study = json.loads((ROOT / "configs/study.json").read_text())
        changed = copy.deepcopy(study)
        changed["training"]["paired_training_seeds"] = [999]
        self.assertEqual(review_configuration(study), review_configuration(changed))
        changed["review_display"]["intensity_window"] = [-100, 200]
        self.assertNotEqual(review_configuration(study), review_configuration(changed))

    def test_exact_pediatric_budget_does_not_require_unused_cases(self):
        rows = [r for r in self.rows if r['case_id'] not in {'pediatric3', 'pediatric4', 'pediatric5'}]
        pools = make_splits(rows, self.settings)
        self.assertEqual(pools['unused_pediatric'], [])
        self.assertIsNone(allocation_counts({'adult': 3, 'pediatric': 3}, 2, 1))
        with self.assertRaisesRegex(ValueError, 'positive integers'):
            allocation_counts({'adult': 20, 'pediatric': 20}, True, 1)

    def test_cohort_swap_rejected_even_without_case_overlap(self):
        pools = make_splits(self.rows, self.settings)
        pools['M0_train'][0], pools['M1_finetune'][0] = pools['M1_finetune'][0], pools['M0_train'][0]
        with self.assertRaisesRegex(ValueError, 'wrong cohort'):
            validate_splits(pools, self.rows, self.settings)

    def test_assessment_cannot_use_test_cases_or_approve_pending_reviews(self):
        case = {'case_id': 'example', 'official_split': 'test', 'cohort': 'pediatric',
                'nonempty_panel_organs': '1', 'nonempty_no_border_organs': '1'}
        with self.assertRaisesRegex(ValueError, 'development cases only'):
            assess_cohorts({'example': case}, {}, {}, ['liver'])
        case['official_split'] = 'train'
        meta = {'example': {'age': '6', 'slice_thickness': '', 'institute': '', 'study_type': '', 'gender': 'f'}}
        reviews = {'example': [{'coverage': 'pending', 'annotation': 'pending'}]}
        result = assess_cohorts({'example': case}, meta, reviews, ['liver'])
        ped = result['profiles']['all_panel_nonempty_no_border']['pediatric']
        self.assertEqual(ped['automatic_candidate_count'], 1)
        self.assertEqual(ped['recorded_full_and_acceptable_all_organs'], 0)
        self.assertEqual(ped['categorical_metadata']['institute'], {'<missing>': 1})


class PreprocessingTests(unittest.TestCase):
    def setup_cache(self, root):
        raw = root / "raw"
        roles = {"adult_train": "M0_train", "pediatric_validation": "pediatric_validation"}
        labels = ["liver", "spleen"]
        rows, organs = {}, {}
        for case, role in roles.items():
            folder = raw / case
            (folder / "segmentations").mkdir(parents=True)
            values = np.arange(512, dtype=np.float32).reshape(8, 8, 8)
            if role != "M0_train":
                values += 100000
            affine = np.diag([1.5, 1.5, 1.5, 1])
            nib.save(nib.Nifti1Image(values, affine), folder / "ct.nii.gz")
            for i, label in enumerate(labels):
                mask = np.zeros((8, 8, 8), dtype=np.uint8)
                mask[1+i:4+i, 1:4, 1:4] = 1
                nib.save(nib.Nifti1Image(mask, affine), folder / "segmentations" / (label + ".nii.gz"))
            audited, checked = audit_case((raw, case, labels, 0.1))
            rows[case] = {"ct_voxel_sha256": audited["ct_voxel_sha256"], "shape": json.dumps(values.shape)}
            organs.update({(case, r["label"]): r for r in checked})
        plan = {"roles": roles, "rows": rows, "labels": labels, "organs": organs,
                "signature": {"expected_native_spacing": [1.5, 1.5, 1.5], "spacing_absolute_tolerance": 0.0001,
                              "split_sha256": "synthetic", "preprocessing": {"sampling_seed": 42, "locations_per_organ": 10}},
                "study_sha256": "synthetic", "provenance": {}}
        args = SimpleNamespace(raw=raw, output=root / "cache", resume=False, limit=1, host_volume=None, reserve_gib=1)
        return args, plan

    def test_cache_resume_round_trip_and_adult_only_normalization(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args, plan = self.setup_cache(Path(tmp))
            run_cache(args, plan)
            first = json.loads((args.output / "manifest.json").read_text())
            self.assertEqual(first["status"], "partial_limit_not_training_ready")
            self.assertFalse((args.output / "normalization.json").exists())
            saved_time = (args.output / "adult_train.npz").stat().st_mtime_ns
            args.resume = True
            run_cache(args, plan)
            result = json.loads((args.output / "manifest.json").read_text())
            self.assertEqual(result["status"], "development_cache_complete_not_training_results")
            self.assertEqual((args.output / "adult_train.npz").stat().st_mtime_ns, saved_time)
            norm = json.loads((args.output / "normalization.json").read_text())
            self.assertEqual(norm["fit_cases"], ["adult_train"])
            self.assertLess(norm["mean"], 512)
            with np.load(args.output / "pediatric_validation.npz") as z:
                self.assertEqual(len(z["normalization_sample"]), 0)
                self.assertEqual(z["data"].dtype, np.float32)
                self.assertEqual(z["seg"].dtype, np.uint8)
                self.assertTrue(np.any(z["seg"] == 3))  # overlapping reference labels are ignored
                self.assertEqual(set(z["foreground_locations"][:, 0]), {1, 2})
            (args.output / "adult_train.npz").write_bytes(b"broken")
            with self.assertRaisesRegex(ValueError, "cache file missing or changed"):
                run_cache(args, plan)

    def test_cache_rejects_contract_changes_and_raw_ct_drift(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args, plan = self.setup_cache(Path(tmp))
            run_cache(args, plan)
            args.resume = True
            changed = copy.deepcopy(plan)
            changed["signature"]["split_sha256"] = "different"
            with self.assertRaisesRegex(ValueError, "signature changed"):
                run_cache(args, changed)
            with self.assertRaisesRegex(ValueError, "CT content differs"):
                preprocess_case(args.raw, "adult_train", plan["labels"], expected_ct_hash="changed")

    def test_disk_guard_checks_host_free_space(self):
        with patch("tsv3.preprocessing.shutil.disk_usage", side_effect=[SimpleNamespace(free=100 * 2**30), SimpleNamespace(free=2 * 2**30)]):
            with self.assertRaisesRegex(OSError, "Insufficient free space"):
                check_space(Path("/cache"), Path("/host"), 1, 10)


class CheckpointTests(unittest.TestCase):
    def test_resume_restores_optimizer_and_random_patch_sequence(self):
        import torch
        from tsv3.training import optimization_components, sample_patch, update_step, save_checkpoint, load_checkpoint, state_hash
        torch.set_num_threads(2)
        torch.manual_seed(42)
        config = {"seed": 42, "patch_size": [4, 4, 4], "foreground_sampling_probability": 1/3,
                  "learning_rate": 0.01, "momentum": 0.99, "weight_decay": 3e-5}
        data = np.arange(512, dtype=np.float32).reshape(1, 8, 8, 8) / 512
        seg = np.zeros_like(data, dtype=np.uint8)
        seg[:, 1:4] = 1
        seg[:, 4:7] = 2
        model = torch.nn.Conv3d(1, 3, 1)
        opt, loss_fn, scaler = optimization_components(model, config, 3, device="cpu")
        rng = np.random.default_rng(42)
        for _ in range(2):
            x, y = sample_patch(data, seg, config, rng, 3, device="cpu")
            update_step(model, x, y, opt, loss_fn, scaler)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resume.pt"
            save_checkpoint(path, model, opt, scaler, rng, 2, config, {"split": "synthetic"})
            x, y = sample_patch(data, seg, config, rng, 3, device="cpu")
            expected_loss, _ = update_step(model, x, y, opt, loss_fn, scaler)
            expected = state_hash(model.state_dict())
            restored = torch.nn.Conv3d(1, 3, 1)
            opt2, loss2, scaler2 = optimization_components(restored, config, 3, device="cpu")
            rng2 = np.random.default_rng(999)
            count = load_checkpoint(path, restored, opt2, scaler2, rng2, config, {"split": "synthetic"})
            self.assertEqual(count, 2)
            x2, y2 = sample_patch(data, seg, config, rng2, 3, device="cpu")
            self.assertTrue(torch.equal(x, x2) and torch.equal(y, y2))
            actual_loss, skipped = update_step(restored, x2, y2, opt2, loss2, scaler2)
            self.assertFalse(skipped)
            self.assertEqual(actual_loss, expected_loss)
            self.assertEqual(state_hash(restored.state_dict()), expected)
            with self.assertRaisesRegex(ValueError, "provenance changed"):
                load_checkpoint(path, restored, opt2, scaler2, rng2, config, {"split": "wrong"})

    def test_overnight_does_not_retry_data_failure(self):
        from tsv3.night import wait_for_writer
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            (cache / "manifest.json").write_text(json.dumps({"status": "failed", "error": "Label preservation failed"}))
            with self.assertRaisesRegex(ValueError, "no automatic retry"):
                wait_for_writer(cache, 1, lambda *args, **kwargs: None)


class TrainingExecutionTests(unittest.TestCase):
    def test_validation_dice_ignore_and_empty_prediction(self):
        from tsv3.training import validation_dice
        target = np.array([0, 1, 1, 2, 3], dtype=np.uint8)
        prediction = np.array([0, 1, 0, 2, 1], dtype=np.uint8)
        np.testing.assert_allclose(validation_dice(prediction, target, 2), [2/3, 1])
        self.assertEqual(validation_dice(np.zeros(5, dtype=np.uint8), target, 2), [0, 0])
        with self.assertRaisesRegex(ValueError, "positive reference"):
            validation_dice(np.zeros(3, dtype=np.uint8), np.array([0, 1, 1]), 2)

    def test_equal_adaptation_budget_and_update_schedule(self):
        from tsv3.training import branch_config, learning_rate_at
        study = json.loads((ROOT / "configs/study.json").read_text())
        one = branch_config(study, "M1", False)
        two = branch_config(study, "M2", False)
        for key in ("optimizer_updates", "learning_rate", "schedule", "augmentation", "foreground_sampling_probability", "patch_size", "batch_size"):
            self.assertEqual(one[key], two[key])
        self.assertEqual(learning_rate_at(one, 0), 0.001)
        self.assertEqual(learning_rate_at(one, 5000), 0)
        self.assertGreater(learning_rate_at(one, 100), learning_rate_at(one, 101))
        with self.assertRaises(ValueError):
            learning_rate_at(one, 5001)

    def test_replication_changes_only_training_seed_and_separates_outputs(self):
        from tsv3.training import branch_config, training_seed, training_output, check_output_seed
        study = json.loads((ROOT / "configs/study.json").read_text())
        unchanged = copy.deepcopy(study)
        seeds = study["training"]["replication_plan"]["seeds"]
        self.assertEqual(training_seed(study), 20261029)
        for branch in ("M0", "M1", "M2"):
            base = branch_config(study, branch, False)
            for seed in seeds:
                config = branch_config(study, branch, False, seed)
                self.assertEqual(config, {**base, "seed": seed})
        self.assertEqual(study, unchanged)
        self.assertEqual(len({training_output(study, s) for s in seeds}), len(seeds))
        self.assertEqual(training_output(study, 20261030).name, "training_abdominal_v1_seed20261030")
        self.assertNotEqual(training_output(study, 20261030, True, True), training_output(study, 20261030))
        with self.assertRaisesRegex(ValueError, "declared seeds"):
            training_seed(study, 999)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            run = {"provenance": {"training_seed": 20261030}, "study_snapshot": study}
            (output / "run.json").write_text(json.dumps(run))
            check_output_seed(output, 20261030)
            with self.assertRaisesRegex(ValueError, "Output belongs to seed"):
                check_output_seed(output, 20261031)
            run["provenance"].pop("training_seed")
            (output / "run.json").write_text(json.dumps(run))
            check_output_seed(output, 20261029)
            with self.assertRaisesRegex(ValueError, "Output belongs to seed"):
                check_output_seed(output, 20261030)

    def test_prefetch_keeps_inputs_rng_and_restart_sequence_exact(self):
        import torch
        from tsv3.training import CasePrefetcher, training_input
        torch.set_num_threads(2)
        study = json.loads((ROOT / "configs/study.json").read_text())
        config = {"patch_size": [4, 4, 4], "foreground_sampling_probability": 1/3,
                  "augmentation": copy.deepcopy(study["training"]["numeric_training_plan"]["augmentation"])}
        for key in ("brightness_probability", "contrast_probability", "noise_probability"):
            config["augmentation"][key] = 1.0
        class Loader:
            def get(self, case):
                n = 3 if case == "small" else 6
                data = np.arange(n**3, dtype=np.float32).reshape(1, n, n, n) / n**3
                seg = np.ones_like(data, dtype=np.uint8)
                return data, seg, np.array([[1, 1, 1, 1]], dtype=np.int32)
        cases = ["small", "large"]
        baseline = []
        rng = np.random.default_rng(123)
        torch.manual_seed(456)
        for _ in range(12):
            case, x, y = training_input(Loader(), cases, config, rng, 2, device="cpu")
            baseline.append((case, x, y, json.dumps(rng.bit_generator.state, sort_keys=True)))
        rng = np.random.default_rng(123)
        torch.manual_seed(456)
        reader = CasePrefetcher(Loader())
        try:
            for index, expected in enumerate(baseline):
                case, x, y = training_input(reader, cases, config, rng, 2, device="cpu")
                self.assertEqual(case, expected[0])
                self.assertTrue(torch.equal(x, expected[1]))
                self.assertTrue(torch.equal(y, expected[2]))
                self.assertEqual(json.dumps(rng.bit_generator.state, sort_keys=True), expected[3])
                reader.prefetch(cases, rng)
                self.assertEqual(json.dumps(rng.bit_generator.state, sort_keys=True), expected[3])
                if index == 5:
                    # A pending speculative read must not affect a checkpoint/restart.
                    saved_rng = copy.deepcopy(rng.bit_generator.state)
                    saved_torch = torch.get_rng_state()
                    reader.close()
                    reader = CasePrefetcher(Loader())
                    rng = np.random.default_rng()
                    rng.bit_generator.state = saved_rng
                    torch.set_rng_state(saved_torch)
        finally:
            reader.close()

    def test_prefetch_rejects_wrong_case_and_propagates_read_failure(self):
        from tsv3.training import CasePrefetcher
        class Loader:
            def get(self, case):
                raise RuntimeError("Cannot decompress: " + case)
        reader = CasePrefetcher(Loader())
        try:
            reader.prefetch(["a"], np.random.default_rng(1))
            with self.assertRaisesRegex(ValueError, "authoritative RNG"):
                reader.get("b")
            with self.assertRaisesRegex(RuntimeError, "Cannot decompress"):
                reader.get("a")
        finally:
            reader.close()

    def test_prefetch_migration_requires_evidence_and_preserves_originals(self):
        from tsv3.training import prepare_prefetch_upgrade
        from tsv3.data import sha256
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "M0").mkdir()
            checkpoint = root / "M0/latest.pt"
            checkpoint.write_bytes(b"original checkpoint")
            old = {"source_sha256": {"training.py": "old"}, "study_sha256": "fixed"}
            new = {**old, "source_sha256": {"training.py": "new"}, "input_pipeline": "case_prefetch_v1"}
            previous = {"status": "interrupted", "research_training": True, "provenance": old}
            (root / "run.json").write_text(json.dumps(previous))
            evidence = root / "evidence.json"
            report = {"status": "throughput_benchmark_complete", "prefetch_recommended": True,
                      "input_equivalence": {"equal": False}, "one_update_weights_close": True,
                      "source_sha256": new["source_sha256"], "original_provenance": old,
                      "checkpoint_sha256": sha256(checkpoint), "run": str(root)}
            evidence.write_text(json.dumps(report))
            with patch("tsv3.training.verify_prefetch_source_change"), patch(
                    "tsv3.training.save_source_snapshot",
                    side_effect=lambda output, name: (output / name).write_bytes(b"snapshot")):
                with self.assertRaisesRegex(ValueError, "Matching equivalence"):
                    prepare_prefetch_upgrade(root, previous, new, evidence)
                self.assertFalse((root / "M0/pre_prefetch_upgrade.pt").exists())
                report["input_equivalence"]["equal"] = True
                evidence.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, "Frozen training"):
                    prepare_prefetch_upgrade(root, previous, {**new, "study_sha256": "changed"}, evidence)
                upgrade = prepare_prefetch_upgrade(root, previous, new, evidence)
            self.assertEqual(checkpoint.read_bytes(), b"original checkpoint")
            self.assertEqual(sha256(root / "M0/pre_prefetch_upgrade.pt"), sha256(checkpoint))
            self.assertEqual(json.loads((root / "run.json").read_text()), previous)
            self.assertEqual(upgrade["old_provenance"], old)

    def test_cache_integrity_and_role_guard(self):
        from tsv3.training import CaseCache
        from tsv3.data import sha256
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "case.npz"
            data = np.arange(64, dtype=np.float32).reshape(1, 4, 4, 4)
            seg = np.zeros_like(data, dtype=np.uint8)
            np.savez_compressed(path, data=data, seg=seg, foreground_locations=np.array([[1, 1, 1, 1]], dtype=np.int32))
            manifest = {"completed": {"case": {"file": path.name, "bytes": path.stat().st_size,
                        "sha256": sha256(path), "properties": {"preprocessed_shape": [4, 4, 4]}}}}
            norm = {"lower": 0, "upper": 63, "mean": 31, "std": 2}
            cache = CaseCache(root, manifest, norm, ["case"], 0.01)
            x, _, _ = cache.get("case")
            np.testing.assert_array_equal(x, (data-31)/2)
            self.assertIs(x, cache.get("case")[0])
            with self.assertRaisesRegex(ValueError, "authorized development"):
                cache.get("test_case")
            path.write_bytes(b"corrupt")
            cache2 = CaseCache(root, manifest, norm, ["case"], 0.01)
            with self.assertRaisesRegex(ValueError, "Cache file changed"):
                cache2.get("case")

    def test_fast_cache_is_lossless_and_preserves_training_input(self):
        from tsv3.data import sha256
        import blosc2
        from tsv3.fast_cache import FORMAT, convert_case, read_case, verify_fast_cache
        from tsv3.training import CaseCache
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original, fast = root / "original", root / "fast"
            original.mkdir()
            fast.mkdir()
            data = np.linspace(-200, 300, 216, dtype=np.float32).reshape(1, 6, 6, 6)
            seg = np.zeros_like(data, dtype=np.uint8)
            seg[:, 2:4] = 1
            locations = np.array([[1, 2, 2, 2]], dtype=np.int32)
            np.savez_compressed(original / "case.npz", data=data, seg=seg, foreground_locations=locations)
            record = {"file": "case.npz", "bytes": (original / "case.npz").stat().st_size,
                      "sha256": sha256(original / "case.npz"), "properties": {"preprocessed_shape": [6, 6, 6]}}
            manifest = {"completed": {"case": record}, "normalization": {"sha256": "fixed-normalization"}}
            (original / "manifest.json").write_text(json.dumps(manifest))
            fast_record = convert_case(original / "case.npz", fast / "case.npz")
            fast_manifest = {"format": FORMAT, "status": "complete", "error": None,
                             "blosc2_version": blosc2.__version__,
                             "source_manifest_sha256": sha256(original / "manifest.json"),
                             "normalization_sha256": "fixed-normalization",
                             "completed": {"case": fast_record}}
            (fast / "manifest.json").write_text(json.dumps(fast_manifest))
            self.assertEqual(set(verify_fast_cache(fast, original, manifest)["completed"]), {"case"})
            fast_manifest["normalization_sha256"] = "wrong"
            (fast / "manifest.json").write_text(json.dumps(fast_manifest))
            with self.assertRaisesRegex(ValueError, "normalization source changed"):
                verify_fast_cache(fast, original, manifest)
            for expected, actual in zip((data, seg, locations), read_case(fast / "case.npz")):
                np.testing.assert_array_equal(expected, actual)
            norm = {"lower": -100, "upper": 200, "mean": 10, "std": 20}
            slow_input = CaseCache(original, manifest, norm, ["case"], 0.01).get("case")
            fast_input = CaseCache(original, manifest, norm, ["case"], 0.01,
                                   fast, {"completed": {"case": fast_record}}).get("case")
            for slow, quick in zip(slow_input, fast_input):
                np.testing.assert_array_equal(slow, quick)
            (fast / "case.npz").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "Cache file changed"):
                CaseCache(original, manifest, norm, ["case"], 0.01,
                          fast, {"completed": {"case": fast_record}}).get("case")

    def test_augmentation_rng_and_targets_resume_exactly(self):
        import torch
        from tsv3.training import augment_patch, save_checkpoint, load_checkpoint, optimization_components
        torch.manual_seed(11)
        config = {"learning_rate": 0.01, "momentum": 0.99, "weight_decay": 3e-5}
        aug = {"brightness_probability": 1, "brightness_range": [0.85, 1.15],
               "contrast_probability": 1, "contrast_range": [0.85, 1.15],
               "noise_probability": 1, "noise_std_range": [0.05, 0.1]}
        model = torch.nn.Conv3d(1, 3, 1)
        optimizer, _, scaler = optimization_components(model, config, 3, device="cpu")
        rng = np.random.default_rng(11)
        x = torch.ones((1, 1, 4, 4, 4))
        y = torch.ones_like(x, dtype=torch.long)
        y[:, :, 0] = 3
        original = y.clone()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resume.pt"
            save_checkpoint(path, model, optimizer, scaler, rng, 0, config, {})
            first, first_y = augment_patch(x, y, aug, rng, 3)
            load_checkpoint(path, model, optimizer, scaler, rng, config, {})
            second, second_y = augment_patch(x, y, aug, rng, 3)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first_y, original) and torch.equal(second_y, original))
        self.assertTrue(torch.all(first[y == 3] == 0))

    def test_foreground_sampling_uses_small_organ_locations(self):
        from tsv3.training import sample_patch
        data = np.zeros((1, 10, 10, 10), dtype=np.float32)
        seg = np.zeros_like(data, dtype=np.uint8)
        seg[0, 7, 7, 7] = 2
        config = {"patch_size": [3, 3, 3], "foreground_sampling_probability": 1}
        _, y = sample_patch(data, seg, config, np.random.default_rng(1), 3, device="cpu",
                            locations=np.array([[2, 7, 7, 7]], dtype=np.int32))
        self.assertEqual(int(y[0, 0, 1, 1, 1]), 2)

    def test_branch_orchestration_initialization_and_resume_on_cpu(self):
        import torch
        from tsv3 import training
        torch.set_num_threads(2)
        study = json.loads((ROOT / "configs/study.json").read_text())
        numeric = study["training"]["numeric_training_plan"]
        numeric["smoke"].update(updates_per_branch=2, checkpoint_every_updates=1, validation_every_updates=2)
        numeric["log_every_updates"] = 1
        study["training"]["fixed_execution_plan"]["patch_size"] = [4, 4, 4]
        partitions = {role: [role] for role in ("M0_train", "M1_finetune", "M2_finetune", "adult_validation", "pediatric_validation")}
        plan = {"study": study, "labels": ["one", "two"], "split": {"partitions": partitions}}
        data = np.arange(216, dtype=np.float32).reshape(1, 6, 6, 6) / 216
        target = np.zeros_like(data, dtype=np.uint8)
        target[:, :3] = 1
        target[:, 3:] = 2
        coordinates = np.array([[1, 1, 1, 1], [2, 4, 4, 4]], dtype=np.int32)
        real_sample, real_optimization = training.sample_patch, training.optimization_components
        validation = {role: {"macro_Dice": 0.5} for role in ("adult_validation", "pediatric_validation")}
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            stack.enter_context(patch("tsv3.night.verify_completion", return_value=({}, {})))
            stack.enter_context(patch.object(training, "build_model", side_effect=lambda *args: torch.nn.Conv3d(1, 3, 1)))
            stack.enter_context(patch.object(training, "sample_patch", side_effect=lambda *args, **kw: real_sample(*args, **{**kw, "device": "cpu"})))
            stack.enter_context(patch.object(training, "optimization_components", side_effect=lambda *args: real_optimization(*args, device="cpu")))
            loader = stack.enter_context(patch.object(training, "CaseCache"))
            loader.return_value.get.return_value = (data, target, coordinates)
            stack.enter_context(patch.object(training, "validate_model", return_value=validation))
            stack.enter_context(patch("torch.cuda.synchronize"))
            stack.enter_context(patch("tsv3.preprocessing.check_space"))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            initial_states = []
            for seed in (20261030, 20261031):
                output = Path(tmp) / str(seed)
                output.mkdir()
                for branch in ("M0", "M1", "M2"):
                    config = training.branch_config(study, branch, True, seed)
                    training.run_branch(branch, plan, output, output, config, {}, output / "M0/selected.pt", False, None,
                                        prefetch=seed == 20261031)
                    first = json.loads((output / branch / "status.json").read_text())
                    self.assertEqual(first["status"], "complete")
                    self.assertEqual(first["seed"], seed)
                    self.assertTrue(first["trainer_state"]["checkpoint_roundtrip_passed"])
                    training.run_branch(branch, plan, output, output, config, {}, output / "M0/selected.pt", True, None,
                                        prefetch=seed == 20261031)
                    self.assertEqual(json.loads((output / branch / "status.json").read_text())["updates"], 2)
                initial_states.append(json.loads((output / "M0/status.json").read_text())["trainer_state"]["initial_state_sha256"])
                a = torch.load(output / "M1/latest.pt", map_location="cpu", weights_only=False)
                b = torch.load(output / "M2/latest.pt", map_location="cpu", weights_only=False)
                self.assertEqual(a["trainer_state"]["initial_state_sha256"], b["trainer_state"]["initial_state_sha256"])
                self.assertEqual(a["provenance"]["M0_checkpoint_sha256"], b["provenance"]["M0_checkpoint_sha256"])
                wrong = training.branch_config(study, "M1", True, 20261029)
                with self.assertRaisesRegex(ValueError, "M0 checkpoint belongs"):
                    training.run_branch("M1", plan, output, output, wrong, {}, output / "M0/selected.pt", True, None)
            self.assertNotEqual(*initial_states)


class FastCacheMaintenanceTests(unittest.TestCase):
    def setup_caches(self, root):
        from tsv3.fast_cache import FORMAT, convert_case
        from tsv3.data import sha256
        import blosc2
        source, fast = root / "source", root / "fast"
        source.mkdir()
        fast.mkdir()
        norm = {"lower": 0, "upper": 63, "mean": 30, "std": 10}
        (source / "normalization.json").write_text(json.dumps(norm))
        original = {"status": "development_cache_complete_not_training_results", "error": None,
                    "signature": {"labels": {"background": 0, "one": 1, "two": 2}},
                    "normalization": {"file": "normalization.json", "sha256": sha256(source / "normalization.json")},
                    "completed": {}}
        records = {}
        for case in ("a", "b"):
            data = np.arange(64, dtype=np.float32).reshape(1, 4, 4, 4)
            seg = np.ones_like(data, dtype=np.uint8)
            seg[:, 2:] = 2
            locations = np.array([[1, 0, 0, 0], [2, 3, 3, 3]], dtype=np.int32)
            path = source / (case + ".npz")
            np.savez_compressed(path, data=data, seg=seg, foreground_locations=locations)
            original["completed"][case] = {"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size,
                                           "properties": {"preprocessed_shape": [4, 4, 4]}}
            records[case] = convert_case(path, fast / path.name)
        (source / "manifest.json").write_text(json.dumps(original))
        record = {"status": "complete", "error": None, "format": FORMAT, "blosc2_version": blosc2.__version__,
                  "source_manifest_sha256": sha256(source / "manifest.json"),
                  "normalization_sha256": original["normalization"]["sha256"], "completed": records}
        (fast / "manifest.json").write_text(json.dumps(record))
        (source / "keep.txt").write_text("unrelated file")
        return source, fast, original, record, norm

    def test_prune_keeps_metadata_and_fast_training_inputs_and_is_repeatable(self):
        from tsv3.fast_cache import verify_and_prune
        from tsv3.training import CaseCache
        from tsv3.data import sha256
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            source, fast, original, record, norm = self.setup_caches(Path(tmp))
            before = CaseCache(source, original, norm, ["a"], 0.01).get("a")
            manifest_hash = sha256(source / "manifest.json")
            report = verify_and_prune(source, fast, True)
            self.assertEqual(report["status"], "verified_source_arrays_pruned")
            self.assertEqual(report["removed_source_cases"], ["a", "b"])
            self.assertFalse(list(source.glob("*.npz")))
            self.assertTrue((source / "keep.txt").exists())
            self.assertEqual(sha256(source / "manifest.json"), manifest_hash)
            after = CaseCache(source, original, norm, ["a"], 0.01, fast, record).get("a")
            for a, b in zip(before, after):
                np.testing.assert_array_equal(a, b)
            repeated = verify_and_prune(source, fast, True)
            self.assertEqual(repeated["source_arrays_already_absent"], 2)
            self.assertEqual(repeated["removed_source_bytes"], 0)

    def test_corrupt_last_fast_file_prevents_all_source_deletions(self):
        from tsv3.fast_cache import verify_and_prune
        with tempfile.TemporaryDirectory() as tmp:
            source, fast, *_ = self.setup_caches(Path(tmp))
            path = fast / "b.npz"
            data = bytearray(path.read_bytes())
            data[len(data) // 2] ^= 1
            path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_and_prune(source, fast, True)
            self.assertEqual(len(list(source.glob("*.npz"))), 2)
            self.assertEqual(json.loads((fast / "verification.json").read_text())["status"], "failed")

    def test_decoder_failure_and_active_reader_prevent_deletion(self):
        from tsv3.fast_cache import verify_and_prune, read_case
        import fcntl
        with tempfile.TemporaryDirectory() as tmp:
            source, fast, *_ = self.setup_caches(Path(tmp))
            def decode(path):
                if path.name == "b.npz":
                    raise RuntimeError("Cannot decompress")
                return read_case(path)
            with patch("tsv3.fast_cache.read_case", side_effect=decode), self.assertRaisesRegex(RuntimeError, "Cannot decompress"):
                verify_and_prune(source, fast, True)
            self.assertEqual(len(list(source.glob("*.npz"))), 2)
            with (source / ".lock").open("a") as reader:
                fcntl.flock(reader, fcntl.LOCK_SH | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "Cache is in use"):
                    verify_and_prune(source, fast, True)
            self.assertEqual(len(list(source.glob("*.npz"))), 2)


class EvaluationTests(unittest.TestCase):
    def test_metrics_identical_shifted_and_empty_masks(self):
        from tsv3.evaluation import score_binary
        reference = np.zeros((14, 14, 14), dtype=bool)
        reference[3:8, 3:8, 3:8] = True
        same = score_binary(reference, reference)
        self.assertEqual((same["Dice"], same["NSD"], same["HD95"]), (1, 1, 0))
        shifted = np.zeros_like(reference)
        shifted[5:10, 3:8, 3:8] = True
        result = score_binary(reference, shifted)
        self.assertAlmostEqual(result["Dice"], 0.6)
        self.assertAlmostEqual(result["HD95"], 2.0)
        self.assertEqual(result["NSD"], 1)
        self.assertLess(score_binary(reference, shifted, 0)["NSD"], result["NSD"])
        self.assertEqual(result, score_binary(shifted, reference))
        self.assertEqual(result, score_binary(reference.transpose(2, 0, 1), shifted.transpose(2, 0, 1)))
        empty = np.zeros_like(reference)
        missing = score_binary(reference, empty)
        self.assertEqual((missing["Dice"], missing["NSD"]), (0, 0))
        self.assertTrue(missing["HD95_infinite"])
        self.assertIsNone(missing["HD95"])
        self.assertIsNone(score_binary(empty, reference)["Dice"])
        self.assertIsNone(score_binary(empty, empty)["Dice"])
        json.dumps(missing, allow_nan=False)
        with self.assertRaisesRegex(ValueError, "boolean masks"):
            score_binary(reference.astype(np.uint8), reference)

    def test_shared_dice_matches_confusion_matrix_and_binary_scores(self):
        from tsv3.evaluation import validation_dice, score_binary
        rng = np.random.default_rng(91)
        for _ in range(5):
            target = rng.integers(0, 4, (8, 8, 8))
            prediction = rng.integers(0, 3, (8, 8, 8))
            actual = validation_dice(prediction, target, 2)
            valid = target != 3
            for label in (1, 2):
                truth = (target == label) & valid
                pred = (prediction == label) & valid
                expected = 2 * np.count_nonzero(truth & pred) / (np.count_nonzero(truth) + np.count_nonzero(pred))
                self.assertEqual(actual[label-1], expected)
                self.assertEqual(score_binary(truth, pred)["Dice"], expected)

    def test_test_eligibility_excludes_empty_and_keeps_border_for_sensitivity(self):
        from tsv3.evaluation import test_eligibility
        split = {"partitions": {"adult_test": ["a"], "pediatric_test": ["p"], "M0_train": ["train"]}}
        cases = [{"case_id": c, "cohort": cohort, "official_split": "test", "automated_eligible": "1"}
                 for c, cohort in (("a", "adult"), ("p", "pediatric"))]
        organs = [{"case_id": c, "label": label, "error": "", "nonzero_voxels": "10", "empty": "0", "touches_border": "0"}
                  for c in ("a", "p") for label in ("one", "two")]
        organs[1].update(nonzero_voxels="0", empty="1")
        organs[3]["touches_border"] = "1"
        result = test_eligibility(split, cases, organs, ["one", "two"])
        self.assertEqual([(r["primary_eligible"], r["sensitivity_eligible"]) for r in result], [(1,1), (0,0), (1,1), (0,1)])
        split["partitions"]["M0_train"].append("p")
        with self.assertRaisesRegex(ValueError, "Development case"):
            test_eligibility(split, cases, organs, ["one", "two"])

    def test_source_snapshot_has_code_and_environment_but_no_dataset(self):
        import tarfile
        from tsv3.training import save_source_snapshot
        from tsv3.data import sha256
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            digest = save_source_snapshot(root)
            self.assertEqual(digest, sha256(root / "source_snapshot.tar.gz"))
            with tarfile.open(root / "source_snapshot.tar.gz") as archive:
                names = archive.getnames()
            self.assertIn("src/tsv3/training.py", names)
            self.assertIn("docs/experiment_log.md", names)
            self.assertIn("configs/study.json", names)
            self.assertIn("uv.lock", names)
            self.assertFalse(any(x.startswith(("data/", "outputs/", ".venv/")) for x in names))
            with self.assertRaises(FileExistsError):
                save_source_snapshot(root)


if __name__ == "__main__":
    unittest.main()
