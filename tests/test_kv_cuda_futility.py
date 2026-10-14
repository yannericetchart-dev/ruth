"""Offline checks: observed CUDA failure trends do not become false hard proof."""
import unittest

from calibration.kv_cuda_futility import (
    _verified_cuda_memory_failure,
    observed_cuda_futility,
)


def _ranking():
    gains = [28] * 3 + [24] * 2 + [8] * 4 + [6] * 3
    return [
        {"tensor": f"blk.{i}.ffn_gate.weight",
         "measured_load_gain_256_mib": gain, "rank": i + 1}
        for i, gain in enumerate(gains)
    ]


def _good_load():
    return {"success": True}


def _cuda_failure():
    return {
        "load_only": _good_load(),
        "inference": {
            "success": False,
            "failure_class": "native_cuda_resource_allocation_failure",
            "samples": [{
                "success": False,
                "failure_class": "native_cuda_resource_allocation_failure",
                "worker_signal": "SIGABRT",
                "failure_diagnostics": {
                    "native_cuda_resource_allocation_failed": True,
                    "failed_during": "prefill",
                },
            }],
        },
    }


def _success():
    return {
        "load_only": _good_load(),
        "inference": {"success": True},
    }


class ObservedFutilityTests(unittest.TestCase):
    def test_stop_only_after_lower_band_native_failures(self):
        ranked = _ranking()
        entries = {
            row["tensor"]: _cuda_failure()
            for row in ranked[:8]
        }
        # Big save-band gap is at index 5 (24 -> 8), so 3 native
        # CUDA failures after that gap suffice in this reduced fixture.
        result = observed_cuda_futility(
            ranked, entries, witness_count=3,
        )
        self.assertIsNotNone(result)
        self.assertTrue(result["stopped"])
        self.assertEqual(result["certified_exclusions"], 0)
        self.assertEqual(result["next_rank"], 9)
        self.assertEqual(result["completed_rank_prefix"], 8)
        self.assertEqual(result["remaining_untested"], 4)
        self.assertIn("non_certified", result["classification"])

    def test_not_enough_lower_band_cuda_failures(self):
        ranked = _ranking()
        entries = {
            row["tensor"]: _cuda_failure()
            for row in ranked[:7]
        }
        self.assertIsNone(observed_cuda_futility(
            ranked, entries, witness_count=3,
        ))

    def test_generic_sigabrt_does_not_qualify(self):
        generic = _cuda_failure()
        generic["inference"]["failure_class"] = "worker_sigabrt"
        generic["inference"]["samples"][0]["failure_class"] = "worker_sigabrt"
        self.assertFalse(_verified_cuda_memory_failure(generic))
        ranked = _ranking()
        entries = {
            row["tensor"]: _cuda_failure()
            for row in ranked[:8]
        }
        entries[ranked[7]["tensor"]] = generic
        self.assertIsNone(observed_cuda_futility(
            ranked, entries, witness_count=3,
        ))

    def test_success_in_lower_band_prevents_incomplete_best_list(self):
        ranked = _ranking()
        entries = {
            row["tensor"]: _cuda_failure()
            for row in ranked[:9]
        }
        entries[ranked[5]["tensor"]] = _success()
        self.assertIsNone(observed_cuda_futility(
            ranked, entries, witness_count=3,
        ))

    def test_success_in_earlier_band_does_not_erase_valid_profile(self):
        ranked = _ranking()
        entries = {
            row["tensor"]: _cuda_failure()
            for row in ranked[:8]
        }
        entries[ranked[0]["tensor"]] = _success()
        result = observed_cuda_futility(
            ranked, entries, witness_count=3,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["certified_exclusions"], 0)
        self.assertEqual(entries[ranked[0]["tensor"]]["inference"]["success"],
                         True)

    def test_incomplete_checkpoint_gap_disables_stop(self):
        ranked = _ranking()
        entries = {
            row["tensor"]: _cuda_failure()
            for row in ranked[:8]
        }
        del entries[ranked[2]["tensor"]]
        self.assertIsNone(observed_cuda_futility(
            ranked, entries, witness_count=3,
        ))

    def test_any_tested_candidate_success_in_new_band_blocks_stop(self):
        ranked = _ranking()
        entries = {
            row["tensor"]: _cuda_failure()
            for row in ranked[:10]
        }
        entries[ranked[6]["tensor"]] = _success()
        self.assertIsNone(observed_cuda_futility(
            ranked, entries, witness_count=3,
        ))


if __name__ == "__main__":
    unittest.main()
