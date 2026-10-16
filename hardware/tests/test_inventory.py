"""Hardware resource reports, only synthetic OS observations."""
import unittest
from unittest.mock import patch
from hardware import DeviceInfo, ResourceReport, HardwareReport, detect, resources

class HardwareInventoryTests(unittest.TestCase):
    def test_ram_available(self):
        with patch("hardware.runtime.inspection.os.sysconf", side_effect=[4096, 1024]) as api:
            ram = resources("system")
        self.assertEqual(ram.total_memory_bytes, 4096 * 1024)
        self.assertEqual(ram.status, "available")
        self.assertEqual(api.call_count, 2)

    def test_ram_unsupported(self):
        with patch("hardware.runtime.inspection.os.sysconf", side_effect=ValueError("missing")):
            self.assertEqual(resources("system").status, "unknown")

    def test_ram_zero(self):
        with patch("hardware.runtime.inspection.os.sysconf", side_effect=[4096, 0]):
            self.assertIsNone(resources().total_memory_bytes)

    def test_ram_negative(self):
        with patch("hardware.runtime.inspection.os.sysconf", side_effect=[-1, 30]):
            self.assertEqual(resources().status, "unknown")

    def test_ram_boolean(self):
        with patch("hardware.runtime.inspection.os.sysconf", side_effect=[True, 20]):
            self.assertEqual(resources().status, "unknown")

    def test_ram_missing_api(self):
        with patch("hardware.runtime.inspection.os.sysconf", side_effect=AttributeError()):
            self.assertEqual(resources().status, "unknown")

    def test_query_rejects_gpu(self):
        with self.assertRaises(ValueError):
            resources("gpu")
        with self.assertRaises(ValueError):
            resources(DeviceInfo("gpu", None, None, "unknown"))

    def test_query_accepts_cpu(self):
        cpu = DeviceInfo("cpu", "synthetic", 4, "available")
        with patch("hardware.runtime.inspection.os.sysconf", side_effect=[1024, 8]):
            self.assertEqual(resources(cpu).total_memory_bytes, 8192)

    def test_inventory_available(self):
        cpu = DeviceInfo("cpu", "synthetic", 4, "available")
        mem = ResourceReport(4096, "available")
        with (patch("hardware.runtime.inspection.detect_cpu", return_value=(cpu,)),
              patch("hardware.runtime.inspection.resources", return_value=mem)):
            report = detect()
        self.assertIsInstance(report, HardwareReport)
        self.assertEqual(report.status, "available")
        self.assertEqual(report.to_dict()["memory"]["total_memory_bytes"], 4096)

    def test_inventory_partial(self):
        with (patch("hardware.runtime.inspection.detect_cpu", return_value=(DeviceInfo("cpu", "synthetic", 4, "available"),)),
              patch("hardware.runtime.inspection.resources", return_value=ResourceReport(None, "unknown"))):
            self.assertEqual(detect().status, "partial")

    def test_inventory_unknown(self):
        with (patch("hardware.runtime.inspection.detect_cpu", return_value=(DeviceInfo("cpu", None, None, "unknown"),)),
              patch("hardware.runtime.inspection.resources", return_value=ResourceReport(None, "unknown"))):
            self.assertEqual(detect().status, "unknown")

    def test_frozen_contract(self):
        report = HardwareReport((DeviceInfo("cpu", None, None, "unknown"),),
                                ResourceReport(None, "unknown"), "unknown")
        with self.assertRaises((AttributeError, TypeError)):
            report.status = "available"

if __name__ == "__main__":
    unittest.main()
