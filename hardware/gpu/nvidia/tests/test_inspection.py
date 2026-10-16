"""Synthetic NVIDIA provider tests, without nvidia-smi or machine probing."""
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from hardware.gpu import GPUDevice, GPUReport, detect
from hardware.gpu.nvidia.runtime.inspection import _parse


class NVIDIAInventoryTests(unittest.TestCase):
    def run_query(self, stdout="", returncode=0, stderr=""):
        with patch("hardware.gpu.nvidia.runtime.inspection.subprocess.run",
                   return_value=SimpleNamespace(stdout=stdout, returncode=returncode, stderr=stderr)) as call:
            result = detect()
        args, kwargs = call.call_args
        self.assertEqual(args[0], ["nvidia-smi", "--query-gpu=name,memory.total",
                                   "--format=csv,noheader,nounits"])
        self.assertIs(kwargs["shell"], False)
        self.assertIs(kwargs["capture_output"], True)
        self.assertGreater(kwargs["timeout"], 0)
        return result

    def test_one_gpu_and_memory_conversion(self):
        report = self.run_query("NVIDIA GeForce RTX 4090, 24564\n")
        self.assertEqual((report.status, report.count), ("available", 1))
        self.assertEqual(report.devices[0].model, "NVIDIA GeForce RTX 4090")
        self.assertEqual(report.devices[0].total_vram_bytes, 24564 * 1024 * 1024)

    def test_two_gpus(self):
        report = self.run_query("NVIDIA A100, 40960\nNVIDIA RTX A5000, 24564\n")
        self.assertEqual(report.count, 2)
        self.assertEqual(report.status, "available")

    def test_empty_success_not_equated_to_no_physical_gpu(self):
        self.assertEqual(self.run_query("").status, "none_reported")

    def test_utility_missing(self):
        with patch("hardware.gpu.nvidia.runtime.inspection.subprocess.run",
                   side_effect=FileNotFoundError("private path")):
            report = detect()
        self.assertEqual((report.status, report.count), ("unavailable", 0))
        self.assertNotIn("private", str(report))

    def test_timeout(self):
        with patch("hardware.gpu.nvidia.runtime.inspection.subprocess.run",
                   side_effect=subprocess.TimeoutExpired(["nvidia-smi"], 5, output=b"private")):
            report = detect()
        self.assertEqual(report.status, "timeout")
        self.assertNotIn("private", str(report))

    def test_driver_or_command_failure(self):
        report = self.run_query(returncode=1, stderr="private serial 123")
        self.assertEqual(report.status, "inspection_failed")
        self.assertNotIn("private", str(report.to_dict()))

    def test_generic_os_failure(self):
        with patch("hardware.gpu.nvidia.runtime.inspection.subprocess.run",
                   side_effect=PermissionError("private")):
            self.assertEqual(detect().status, "inspection_failed")

    def test_malformed_rows_are_not_reported_as_gpu(self):
        self.assertEqual(self.run_query("unexpected output").status, "unknown")

    def test_malformed_row_with_one_valid_marks_partial(self):
        report = self.run_query("NVIDIA A100, 40960\nnoise\n")
        self.assertEqual((report.count, report.status), (1, "partial"))

    def test_missing_vram_marks_partial_not_zero(self):
        report = self.run_query("NVIDIA A100, N/A\n")
        self.assertEqual(report.status, "partial")
        self.assertIsNone(report.devices[0].total_vram_bytes)

    def test_missing_model_marks_partial(self):
        report = self.run_query(", 24564\n")
        self.assertEqual(report.status, "partial")
        self.assertIsNone(report.devices[0].model)

    def test_invalid_negative_or_zero_vram(self):
        self.assertIsNone(self.run_query("NVIDIA A100, -1").devices[0].total_vram_bytes)
        self.assertIsNone(self.run_query("NVIDIA A100, 0").devices[0].total_vram_bytes)

    def test_control_characters_are_never_propagated(self):
        report = self.run_query("private\x1b[31m, 40960")
        self.assertIsNone(report.devices[0].model)
        self.assertNotIn("private", str(report))

    def test_limit_output_size(self):
        self.assertEqual(_parse("x" * 65537).status, "unknown")

    def test_limit_device_count(self):
        self.assertEqual(_parse("NVIDIA GPU, 2048\n" * 33).status, "unknown")

    def test_frozen_public_report(self):
        report = GPUReport((GPUDevice("NVIDIA", "synthetic", 1024, "available"),), "available")
        with self.assertRaises((AttributeError, TypeError)):
            report.status = "unknown"
        self.assertEqual(report.to_dict()["devices"][0]["vendor"], "NVIDIA")


if __name__ == "__main__":
    unittest.main()
