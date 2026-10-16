"""CPU inventory: synthetic observations, never host-dependent."""
import unittest
from unittest.mock import patch
from hardware.cpu import detect

class CpuInventoryTests(unittest.TestCase):
    def observe(self, arch, count):
        with (patch("hardware.cpu.runtime.inspection.platform.machine", return_value=arch),
              patch("hardware.cpu.runtime.inspection.os.cpu_count", return_value=count)):
            return detect()[0]

    def test_complete(self):
        self.assertEqual((self.observe("x86_64", 8).architecture,
                          self.observe("x86_64", 8).status), ("x86_64", "available"))

    def test_unknown(self):
        self.assertEqual(self.observe("", None).status, "unknown")

    def test_architecture_only(self):
        self.assertEqual(self.observe("arm64", None).status, "partial")

    def test_count_only(self):
        self.assertEqual(self.observe(None, 2).status, "partial")

    def test_invalid_count(self):
        self.assertIsNone(self.observe("", True).logical_processors)
        self.assertIsNone(self.observe("", -1).logical_processors)

    def test_trimmed_architecture(self):
        self.assertEqual(self.observe(" x86_64 ", 4).architecture, "x86_64")

    def test_architecture_failure(self):
        with (patch("hardware.cpu.runtime.inspection.platform.machine", side_effect=OSError()),
              patch("hardware.cpu.runtime.inspection.os.cpu_count", return_value=None)):
            self.assertEqual(detect()[0].status, "unknown")

    def test_processor_failure(self):
        with (patch("hardware.cpu.runtime.inspection.platform.machine", return_value="arm64"),
              patch("hardware.cpu.runtime.inspection.os.cpu_count", side_effect=OSError())):
            self.assertEqual(detect()[0].status, "partial")

if __name__ == "__main__":
    unittest.main()
