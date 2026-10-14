"""Offline regression for import-corrupted native CUDA checkpoint recovery."""
import unittest

from calibration.kv_worker_infrastructure import (
    recover_worker_import_failures, worker_import_failure,
)


def bad_load(module="calibration.kv_single_to_512"):
    return {
        "success": False,
        "failure_class": "load_or_native_verification_failed",
        "failure_diagnostics": {
            "stderr_tail": (
                "/home/yann/Dev/ruth/.venv/bin/python: "
                "No module named " + module + "\n"
            ),
        },
    }


def bad_inference():
    return {
        "success": False,
        "failure_class": "worker_protocol_error",
        "samples": [{
            "success": False, "worker_returncode": 1,
            "failure_class": "worker_protocol_error",
            "failure_diagnostics": {
                "stderr_tail": (
                    "/home/yann/Dev/ruth/.venv/bin/python: "
                    "No module named calibration.mapping_phase\n"
                ),
            },
        }],
    }


class WorkerInfrastructureTests(unittest.TestCase):
    def test_real_checkpoint_python_import_patterns(self):
        self.assertTrue(worker_import_failure(bad_load()))
        self.assertTrue(worker_import_failure(bad_inference()))
        self.assertFalse(worker_import_failure({
            "success": False,
            "failure_class": "native_cuda_oom",
            "samples": [{
                "success": False,
                "failure_diagnostics": {
                    "stderr_tail": "CUDA error: out of memory\\n"
                },
            }],
        }))
        self.assertFalse(worker_import_failure({
            "success": False, "failure_class": "native_cuda_oom",
            "failure_diagnostics": {
                "stderr_tail": "RUTH_WORKER_PHASE=prefill_begin\\n"
            },
        }))

    def test_repair_only_import_errors_preserving_native_cuda_and_history(self):
        entries = {
            "blk.20.ffn_gate.weight": {
                "load_only": {"success": True},
                "inference": {
                    "success": False,
                    "failure_class": "native_cuda_oom",
                    "samples": [{
                        "success": False,
                        "failure_class": "native_cuda_oom",
                        "failure_diagnostics": {
                            "stderr_tail": "CUDA error: out of memory",
                        },
                    }],
                },
            },
            "blk.21.ffn_up.weight": {
                "load_only": {"success": True},
                "inference": bad_inference(),
            },
            "blk.22.ffn_up.weight": {
                "load_only": bad_load(),
            },
        }
        genuine = entries["blk.20.ffn_gate.weight"]["inference"]
        repaired = recover_worker_import_failures(entries)
        self.assertEqual(repaired["recovered_load_only_count"], 1)
        self.assertEqual(repaired["recovered_inference_count"], 1)
        self.assertIs(entries["blk.20.ffn_gate.weight"]["inference"], genuine)
        self.assertNotIn("inference", entries["blk.21.ffn_up.weight"])
        self.assertNotIn("load_only", entries["blk.22.ffn_up.weight"])
        self.assertTrue(worker_import_failure(
            entries["blk.21.ffn_up.weight"]["inference_history"][0],
        ))
        self.assertTrue(worker_import_failure(
            entries["blk.22.ffn_up.weight"]["load_only_history"][0],
        ))
        self.assertEqual(recover_worker_import_failures(entries)[
            "recovered_load_only_count"
        ], 0)


if __name__ == "__main__":
    unittest.main()
