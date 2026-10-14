"""Stage-2 singleton-at-512 tests without a physical GPU."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from calibration.kv_single_to_512 import (
    order_single_candidates, run_single_to_512, probe_load_only
)
from calibration.kv_single_tensor_census import run_single_tensor_census
from tests.test_kv_single_tensor_census import prepare
from tests.test_kv_adaptive_tensor_offload import fake_native_probe


def make_census(root):
    _, layout, greedy_path = prepare(root)
    census = root / "census.json"
    with patch("calibration.kv_single_tensor_census.inspect_gguf",
               return_value=layout):
        complete = run_single_tensor_census(
            source_report=greedy_path, output=census,
            probe=fake_native_probe,
        )
    if not complete["completed"]:
        raise AssertionError("Fixture census must be complete.")
    return layout, census


def fake_load(path, model_id, trial, *, n_ctx, timeout):
    assert n_ctx == 512
    names = trial["host_tensor_names"]
    # Deliberately load an otherwise low-gain singleton and reject a big one.
    ok = (not names or names == ["blk.0.attn_norm.weight"])
    peak = 6100 if ok else 6138
    return {
        "success": ok, "n_ctx": n_ctx,
        "host_tensor_names": names, "mapping_id": trial["id"],
        "native_placement_verified": ok,
        "gpu_kv_verified": ok,
        "native_kv_type_verified": ok,
        "failure_class": None if ok else "native_cuda_oom",
        "measurements": {
            "vram_before_mib": 300., "vram_after_load_mib": peak-30,
            "vram_peak_load_only_mib": peak,
            "vram_load_peak_increment_mib": peak-300,
            "vram_total_mib": 6141.,
        },
    }


def fake_infer(path, model_id, trial, *, n_ctx, tokens, timeout,
               flash_attn, type_k, type_v):
    assert n_ctx == 512
    assert tokens == 16
    assert flash_attn is False and type_k == type_v == "f16"
    return fake_native_probe(
        path, model_id, trial, n_ctx=n_ctx,
        tokens=tokens, timeout=timeout,
        flash_attn=flash_attn, type_k=type_k, type_v=type_v
    )


class SingleTo512Tests(unittest.TestCase):
    def test_rank_uses_verified_post_load_256_vram_without_hard_exclusion(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            layout, census_path = make_census(root)
            source = json.loads(census_path.read_text())
            ranked = order_single_candidates(source, target_ctx=512)
            self.assertEqual(len(ranked), 6)
            self.assertEqual([x["rank"] for x in ranked], list(range(1, 7)))
            self.assertTrue(all(x["hard_excluded"] is False for x in ranked))
            self.assertEqual(ranked[0]["measured_load_gain_256_mib"], 40.)
            self.assertEqual(ranked[-1]["measured_load_gain_256_mib"], 6.)
            self.assertEqual(
                [x["measured_load_gain_256_mib"] for x in ranked],
                sorted((x["measured_load_gain_256_mib"] for x in ranked),
                       reverse=True)
            )

    def test_peak_inference_cannot_override_post_load_savings(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, census_path = make_census(root)
            source = json.loads(census_path.read_text())
            # attn_norm saves just 6 MiB after loading, attn_k 15 MiB.
            # Forge a much lower inference peak for norm: its position
            # MUST remain below attn_k because this phase is about load.
            source["tensor_results"]["blk.0.attn_norm.weight"][
                "metrics"]["vram_peak_mib"] = 4900.
            ranked = order_single_candidates(source, target_ctx=512)
            names = [r["tensor"] for r in ranked]
            self.assertLess(
                names.index("blk.0.attn_k.weight"),
                names.index("blk.0.attn_norm.weight"),
            )
            self.assertEqual(
                ranked[0]["measured_load_gain_256_mib"], 40.
            )

    def test_missing_or_unverified_native_load_samples_are_not_silently_used(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, census_path = make_census(root)
            source = json.loads(census_path.read_text())
            name = "blk.0.ffn_down.weight"
            source["reused_from_greedy"][name]["samples"][0][
                "vram_after_load_mib"] = None
            with self.assertRaisesRegex(ValueError, "vram_after_load"):
                order_single_candidates(source, target_ctx=512)

    def test_load_only_precedes_benchmark_and_negative_load_skips_inference(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            layout, census = make_census(root)
            output = root / "phase2.json"
            events = []
            def loading(*args, **kwargs):
                events.append(("load", tuple(args[2]["host_tensor_names"])))
                return fake_load(*args, **kwargs)
            def inference(*args, **kwargs):
                events.append(("infer", tuple(args[2]["host_tensor_names"])))
                return fake_infer(*args, **kwargs)
            with patch("calibration.kv_single_to_512.inspect_gguf",
                       return_value=layout):
                result = run_single_to_512(
                    census_path=census, output=output, target_ctx=512,
                    load_probe=loading, inference_probe=inference,
                )
            self.assertTrue(result["completed"])
            self.assertEqual(result["summary"]["candidates"], 6)
            self.assertEqual(result["summary"]["load_attempted"], 6)
            self.assertEqual(result["summary"]["load_success"], 1)
            self.assertEqual(result["summary"]["inference_attempted"], 1)
            self.assertEqual(result["summary"]["inference_success"], 1)
            self.assertEqual(events[0], ("load", ()))
            self.assertEqual(len([e for e in events if e[0] == "load"]), 7)
            self.assertEqual(len([e for e in events if e[0] == "infer"]), 2)
            self.assertEqual(events[-2:],
                             [("infer", ("blk.0.attn_norm.weight",))]*2)
            self.assertEqual(
                result["results"]["blk.0.attn_norm.weight"][
                    "load_only"]["measurements"]["vram_load_peak_increment_mib"],
                5800.,
            )
            self.assertEqual(result["hard_pruning"]["count"], 0)
            self.assertEqual(
                result["summary"]["minimum_observed_successful_load_peak_mib"],
                6100.,
            )
            self.assertEqual(
                result["summary"]["minimum_observed_successful_load_increment_mib"],
                5800.,
            )

    def test_resume_between_load_and_inference_does_not_repeat_successful_load(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            layout, census = make_census(root)
            out = root / "stage2.json"
            calls = []
            def load(*args, **kwargs):
                calls.append(("load", tuple(args[2]["host_tensor_names"])))
                return fake_load(*args, **kwargs)
            def infer(*args, **kwargs):
                calls.append(("infer", tuple(args[2]["host_tensor_names"])))
                return fake_infer(*args, **kwargs)
            with patch("calibration.kv_single_to_512.inspect_gguf",
                       return_value=layout):
                first = run_single_to_512(
                    census_path=census, output=out,
                    max_new_loads=1,
                    max_new_benchmarks=1,
                    load_probe=load, inference_probe=infer,
                )
                self.assertFalse(first["completed"])
                calls_before = list(calls)
                final = run_single_to_512(
                    census_path=census, output=out,
                    load_probe=load, inference_probe=infer,
                )
                self.assertTrue(final["completed"])
                self.assertEqual(
                    len([x for x in calls if x[0] == "load"]), 7
                )
                self.assertTrue(
                    calls[:len(calls_before)] == calls_before
                )
                repeat = run_single_to_512(
                    census_path=census, output=out,
                    load_probe=load, inference_probe=infer,
                )
                self.assertTrue(repeat["completed"])
                self.assertEqual(len(calls), 9)

    def test_replay_failed_inference_archives_sigabrt_without_new_loads(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            layout, census = make_census(root)
            output = root / "replay.json"
            loaded = []
            def loading(*args, **kwargs):
                loaded.append(tuple(args[2]["host_tensor_names"]))
                return fake_load(*args, **kwargs)
            calls = []
            def failed_inference(*args, **kwargs):
                calls.append("failed")
                return {
                    "success": False,
                    "failure_class": "native_cuda_resource_allocation_failure",
                    "worker_signal": "SIGABRT",
                    "worker_returncode": -6,
                    "stderr_tail": "CUDA error: the resource allocation failed",
                    "last_worker_phase": "prefill_begin",
                    "failed_during": "prefill",
                }
            with patch("calibration.kv_single_to_512.inspect_gguf",
                       return_value=layout):
                first = run_single_to_512(
                    census_path=census, output=output,
                    load_probe=loading, inference_probe=failed_inference,
                )
                self.assertTrue(first["completed"])
                count_before = len(loaded)
                replay = run_single_to_512(
                    census_path=census, output=output,
                    retry_failed_inferences=1, retry_only=True,
                    load_probe=lambda *a, **kw: self.fail(
                        "Replay must never repeat load-only."
                    ),
                    inference_probe=fake_infer,
                )
            self.assertEqual(len(loaded), count_before)
            self.assertTrue(replay["completed"])
            self.assertEqual(len(calls), 1)
            entry = replay["results"]["blk.0.attn_norm.weight"]
            self.assertEqual(len(entry["inference_history"]), 1)
            self.assertFalse(entry["inference_history"][0]["success"])
            self.assertEqual(
                entry["inference_history"][0]["samples"][0]["worker_signal"],
                "SIGABRT"
            )
            self.assertTrue(entry["inference"]["success"])

    def test_no_load_success_still_finishes_for_phase3(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            layout, census = make_census(root)
            def reject(*args, **kwargs):
                row = fake_load(*args, **kwargs)
                row["success"] = False
                row["failure_class"] = "native_cuda_oom"
                return row
            with patch("calibration.kv_single_to_512.inspect_gguf",
                       return_value=layout):
                report = run_single_to_512(
                    census_path=census, output=root / "none.json",
                    load_probe=reject,
                    inference_probe=lambda *args, **kw: self.fail(
                        "Inferenza vietata dopo load fallito."
                    ),
                )
            self.assertTrue(report["completed"])
            self.assertEqual(report["status"], "completed_no_singleton_inference")
            self.assertEqual(report["summary"]["inference_attempted"], 0)
            self.assertTrue(report["summary"]["ready_for_phase3"])

    def test_never_silently_skip_missing_census_data(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            layout, census = make_census(root)
            data = json.loads(census.read_text())
            data["tensor_results"].pop("blk.0.attn_norm.weight")
            with self.assertRaisesRegex(ValueError, "Censimento non completo"):
                order_single_candidates(data, target_ctx=512)


if __name__ == "__main__":
    unittest.main()
