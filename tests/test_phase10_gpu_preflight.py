"""Offline NVIDIA GPU preflight tests; no CUDA, GPU, or display-manager calls."""
import subprocess
import unittest
from types import SimpleNamespace

from ruth_runtime.gpu_preflight import parse_gpu_xml, inspect_gpu_preflight

UUID = "GPU-5a841828-882b-85d9-c98e-3a0d29e66e00"


def snapshot(processes=""):
    return f"""<?xml version="1.0"?>
<nvidia_smi_log>
  <driver_version>595.99.02</driver_version>
  <gpu>
    <uuid>{UUID}</uuid>
    <fb_memory_usage>
      <total>6141 MiB</total><reserved>369 MiB</reserved>
      <used>15 MiB</used><free>5759 MiB</free>
    </fb_memory_usage>
    <processes>{processes}</processes>
  </gpu>
</nvidia_smi_log>"""


XORG = """<process_info>
<pid>3032</pid><type>G</type><process_name>/usr/lib/xorg/Xorg</process_name>
<used_memory>4 MiB</used_memory>
</process_info>"""


class GpuPreflightTests(unittest.TestCase):
    def test_graphics_process_refuses_probe_and_advises_tty(self):
        report = parse_gpu_xml(snapshot(XORG), UUID)
        self.assertEqual(report.status, "gpu_busy")
        self.assertEqual(report.processes[0]["pid"], 3032)
        self.assertEqual(report.processes[0]["type"], "G")
        self.assertEqual(report.processes[0]["gpu_memory_mib"], 4)
        self.assertIn("TTY", " ".join(report.recommendations))
        self.assertEqual(report.memory_mib["reserved"], 369)
        self.assertEqual(report.memory_mib["free"], 5759)
        self.assertEqual(report.vram_class_gib, 6)

    def test_empty_gpu_allows_probe(self):
        report = parse_gpu_xml(snapshot(), UUID)
        self.assertEqual(report.status, "gpu_ready")
        self.assertFalse(report.processes)
        self.assertFalse(report.recommendations)

    def test_compute_and_graphics_processes(self):
        compute = XORG.replace("3032", "8000").replace("<type>G</type>", "<type>C</type>").replace("/usr/lib/xorg/Xorg", "python")
        report = parse_gpu_xml(snapshot(XORG + compute), UUID)
        self.assertEqual([p["pid"] for p in report.processes], [3032, 8000])
        self.assertEqual(report.status, "gpu_busy")

    def test_missing_process_telemetry_fails_closed(self):
        with self.assertRaises(ValueError):
            parse_gpu_xml(snapshot().replace("<processes></processes>", ""), UUID)

    def test_wrong_gpu_identity_fails_closed(self):
        with self.assertRaises(ValueError):
            parse_gpu_xml(snapshot(), "GPU-aaaaaaaa-1111-2222-3333-bbbbbbbbbbbb")

    def test_command_is_read_only_and_bound_to_uuid(self):
        def fake_run(args, **kw):
            self.assertEqual(args, ["nvidia-smi", "-q", "-x", "-i", UUID])
            self.assertTrue(kw["check"])
            return subprocess.CompletedProcess(args, 0, snapshot(XORG), "")
        report = inspect_gpu_preflight(UUID, run=fake_run)
        self.assertEqual(report.status, "gpu_busy")

    def test_unavailable_nvidia_smi_blocks_gpu(self):
        def unavailable(*args, **kw):
            raise FileNotFoundError("nvidia-smi")
        report = inspect_gpu_preflight(UUID, run=unavailable)
        self.assertEqual(report.status, "gpu_preflight_unverified")
        self.assertTrue(report.error)
        self.assertTrue(report.recommendations)


if __name__ == "__main__":
    unittest.main()
