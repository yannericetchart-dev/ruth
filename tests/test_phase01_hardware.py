"""CPU-only contract tests; fixture GPU != verified runtime support."""
from pathlib import Path
import json
import subprocess
import tempfile
import unittest

from ruth_runtime.hardware import (
    inspect_linux_hardware, normalize_pci_address, parse_cpuinfo,
    parse_meminfo, parse_nvidia_smi, LinuxHardwareProvider,
)

CPUINFO = """processor : 0
vendor_id : GenuineIntel
model name : Example Intel CPU
physical id : 0
core id : 0
flags : sse4_2 avx avx2

processor : 1
vendor_id : GenuineIntel
model name : Example Intel CPU
physical id : 0
core id : 0
flags : sse4_2 avx avx2

processor : 2
vendor_id : GenuineIntel
model name : Example Intel CPU
physical id : 0
core id : 1
flags : avx avx2
"""
MEMINFO = "MemTotal: 16384000 kB\nMemAvailable: 9216000 kB\nSwapTotal: 2048000 kB\n"
NVIDIA = 'GPU-abc, 00000000:01:00.0, "NVIDIA RTX, test", 6144, 4096, 570.20\n'


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.proc = base / "proc"
        self.drm = base / "drm"
        self.proc.mkdir()
        self.drm.mkdir()
        (self.proc / "cpuinfo").write_text(CPUINFO)
        (self.proc / "meminfo").write_text(MEMINFO)

    def gpu(self, card: int, vendor: str, pci: str):
        dev = self.drm / f"card{card}" / "device"
        dev.mkdir(parents=True)
        (dev / "vendor").write_text(vendor)
        (dev / "uevent").write_text("PCI_SLOT_NAME=" + pci + "\n")

    def inspect(self, stdout=NVIDIA, system="Linux", architecture="x86_64",
                failure=None):
        def fake_run(argv, **kwargs):
            self.assertEqual(argv[0], "nvidia-smi")
            self.assertTrue(kwargs["check"])
            if failure is not None:
                raise failure
            return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")
        return inspect_linux_hardware(
            proc_root=self.proc, drm_root=self.drm, run=fake_run,
            system=system, architecture=architecture, release="6.7.0",
            logical_processors=3,
        )


class HardwareTests(Fixture):
    def test_cpu_parsing_has_no_vendor_assumption(self):
        parsed = parse_cpuinfo(CPUINFO, arch="x86_64", logical=3)
        self.assertEqual((parsed.vendor, parsed.logical_processors, parsed.physical_cores),
                         ("intel", 3, 2))
        self.assertIn("avx2", parsed.flags)
        amd = parse_cpuinfo(CPUINFO.replace("GenuineIntel", "AuthenticAMD"),
                            arch="x86_64")
        self.assertEqual(amd.vendor, "amd")

    def test_memory_kib_converted_to_bytes_unknown_not_invented(self):
        m = parse_meminfo(MEMINFO)
        self.assertEqual(m.total_bytes, 16384000 * 1024)
        self.assertEqual(m.available_bytes, 9216000 * 1024)
        self.assertIsNone(parse_meminfo("MemTotal: 100 kB\n").available_bytes)

    def test_pci_normalization_keeps_domain_and_rejects_ambiguity(self):
        self.assertEqual(normalize_pci_address("00000000:01:00.0"), "0000:01:00.0")
        self.assertEqual(normalize_pci_address("0000:0A:00.1"), "0000:0a:00.1")
        self.assertIsNone(normalize_pci_address("00010000:01:00.0"))
        self.assertIsNone(normalize_pci_address("bad"))

    def test_nvidia_parser_handles_csv_quotes_and_missing_values(self):
        rows = parse_nvidia_smi(
            NVIDIA + "GPU-other, 00000000:02:00.0, Other, N/A, N/A, N/A\n"
        )
        self.assertEqual(rows[0].name, "NVIDIA RTX, test")
        self.assertEqual(rows[0].free_bytes, 4096 * 1024**2)
        self.assertIsNone(rows[1].total_bytes)
        self.assertIsNone(rows[1].driver)

    def test_universal_sysfs_and_nvidia_pci_merge(self):
        self.gpu(0, "0x10de", "0000:01:00.0")
        self.gpu(1, "0x1002", "0000:02:00.0")
        self.gpu(2, "0x8086", "0000:00:02.0")
        report = self.inspect()
        self.assertTrue(report.supported_platform)
        self.assertEqual(len(report.gpus), 3)
        by_vendor = {d.vendor: d for d in report.gpus}
        self.assertEqual(set(by_vendor), {"nvidia", "amd", "intel"})
        self.assertEqual(by_vendor["nvidia"].uuid, "GPU-abc")
        self.assertEqual(by_vendor["nvidia"].driver, "570.20")
        self.assertEqual(by_vendor["nvidia"].total_memory_bytes, 6144 * 1024**2)
        self.assertIsNone(by_vendor["amd"].total_memory_bytes)
        self.assertIsNone(by_vendor["intel"].available_memory_bytes)
        self.assertNotIn("cuda", json.dumps(report.to_dict()).lower().replace(
            "cuda_visible_devices", ""
        ))

    def test_unmatched_nvidia_remains_distinct(self):
        self.gpu(0, "0x10de", "0000:02:00.0")
        report = self.inspect()
        self.assertEqual(len(report.gpus), 2)
        self.assertEqual(sum(d.total_memory_bytes is not None for d in report.gpus), 1)
        self.assertIsNone(next(d for d in report.gpus
                               if d.pci_address == "0000:02:00.0").uuid)

    def test_nvidia_smi_failure_preserves_sysfs_and_unknown_memory(self):
        self.gpu(0, "0x10de", "0000:01:00.0")
        report = self.inspect(failure=FileNotFoundError())
        self.assertEqual(len(report.gpus), 1)
        self.assertIsNone(report.gpus[0].available_memory_bytes)
        self.assertTrue(any("telemetria" in w for w in report.warnings))

    def test_non_linux_or_wrong_arch_not_reported_supported(self):
        for system, arch in (("Darwin", "x86_64"), ("Linux", "aarch64")):
            with self.subTest(system=system, arch=arch):
                report = self.inspect(system=system, architecture=arch)
                self.assertFalse(report.supported_platform)
                self.assertFalse(report.gpus)

    def test_headless_nvidia_device_detected_with_pci(self):
        report = self.inspect()
        self.assertEqual(len(report.gpus), 1)
        self.assertEqual(report.gpus[0].pci_address, "0000:01:00.0")

    def test_nvidia_without_pci_does_not_falsely_attach(self):
        self.gpu(0, "0x10de", "0000:01:00.0")
        report = self.inspect(stdout="GPU-abc, N/A, RTX, 6144, 4096, 570.20\n")
        self.assertEqual(len(report.gpus), 1)
        self.assertIsNone(report.gpus[0].uuid)
        self.assertTrue(any("nessun merge" in w for w in report.warnings))

    def test_invalid_gpu_pci_keeps_honest_local_identifier(self):
        self.gpu(0, "0x1002", "some-other-bus")
        report = self.inspect(stdout="")
        self.assertEqual(report.gpus[0].identifier, "drm:card0")
        self.assertIsNone(report.gpus[0].pci_address)

    def test_provider_matches_protocol_shape(self):
        self.assertTrue(callable(getattr(LinuxHardwareProvider(), "discover", None)))
        self.assertTrue(callable(getattr(LinuxHardwareProvider(), "inspect", None)))

    def test_no_cpu_or_memory_proc_does_not_invent_data(self):
        (self.proc / "cpuinfo").unlink()
        (self.proc / "meminfo").unlink()
        report = self.inspect(stdout="")
        self.assertIsNone(report.memory.total_bytes)
        self.assertEqual(report.cpu.vendor, "unknown")
        self.assertTrue(any("cpuinfo" in w for w in report.warnings))


if __name__ == "__main__":
    unittest.main()
