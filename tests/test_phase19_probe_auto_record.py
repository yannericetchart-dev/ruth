"""Regression coverage for automatic native probe evidence persistence and Git sync.

No CUDA imports, GGUF loads, network calls or actual Git operations.
"""
import json
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from ruth_runtime.launcher import main
from ruth_runtime.probe_profile_sync import record_probe_evidence


SHA = "13b2a7b4115bbd0900162edcebe476da1ba1fc24e718e8b40d32f6e300f56dfe"


def native_report(status="gpu_load_only_completed"):
    return {
        "action": "models", "status": status,
        "candidate": "gemma-2-9B-it/gguf/Q4_K_M",
        "mode": "context", "model_verified": True,
        "model_sha256": SHA,
        "native_binding_version": "0.3.36",
        "gpu_uuid": "GPU-native-worker-observation",
        "gpu_preflight": {
            "status": "gpu_ready", "processes": [],
            "gpu_uuid": "GPU-native-worker-observation",
            "driver_version": "595.99.02",
            "memory_mib": {"free": 5773, "total": 6141, "reserved": 369},
        },
        "native_buffers": [
            {"device": "CUDA0", "kind": "model", "mib": 5488.40},
            {"device": "CUDA0", "kind": "kv", "mib": 42.0},
            {"device": "CUDA0", "kind": "kv", "mib": 42.0},
            {"device": "CUDA0", "kind": "compute", "mib": 63.38},
        ],
        "weights_loaded": True, "context_initialized": True,
        "inference_executed": False, "generated_tokens": 0,
        "cpu_fallback_selected": False,
    }


class NativeAutoRecordingTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.profile = self.root / ".ruth-build/profiles/models" / (SHA + ".json")

    def test_success_persists_original_report_before_sanitized_owner_git_sync(self):
        with (patch("ruth_runtime.profile_registry.record_local_profile",
                    return_value={
                        "status": "local_profile_updated",
                        "profile_file": str(self.profile), "model_sha256": SHA,
                    }) as recorder,
              patch("ruth_runtime.profile_contribution.build_proposal",
                    return_value={"proposal": {"safe": True}}) as builder,
              patch("ruth_runtime.profile_contribution.sync_owner_profile_if_authenticated",
                    return_value={"status": "git_profile_updated_model_profiling",
                                  "git_updated": True}) as publisher):
            result = record_probe_evidence(native_report(), project_root=self.root)
        self.assertEqual(result["status"], "local_profile_updated")
        self.assertTrue(result["runtime_profile_published"])
        archived = Path(result["local_evidence_file"])
        self.assertTrue(archived.is_file())
        self.assertEqual(json.loads(archived.read_text()), native_report())
        self.assertIn(str(self.root / ".ruth-build/profiles/evidence"), str(archived))
        self.assertEqual(recorder.call_args.kwargs["project_root"], self.root)
        self.assertEqual(recorder.call_args.args[0], archived)
        self.assertEqual(builder.call_args.args[0], self.profile)
        self.assertEqual(publisher.call_count, 1)

    def test_no_git_sync_still_records_locally(self):
        with (patch("ruth_runtime.profile_registry.record_local_profile",
                    return_value={
                        "status": "local_profile_updated",
                        "profile_file": str(self.profile),
                    }) as recorder,
              patch("ruth_runtime.profile_contribution.sync_owner_profile_if_authenticated",
                    side_effect=AssertionError("Must not publish"))):
            result = record_probe_evidence(
                native_report(), project_root=self.root, sync_git=False)
        self.assertEqual(result["git_publication"]["status"], "git_sync_disabled")
        self.assertTrue(Path(result["local_evidence_file"]).is_file())
        self.assertFalse(result["runtime_profile_published"])
        recorder.assert_called_once()

    def test_failure_still_preserves_local_evidence_even_if_git_unavailable(self):
        with (patch("ruth_runtime.profile_registry.record_local_profile",
                    return_value={
                        "status": "local_profile_updated",
                        "profile_file": str(self.profile),
                    }),
              patch("ruth_runtime.profile_contribution.build_proposal",
                    return_value={"proposal": {"safe": True}}),
              patch("ruth_runtime.profile_contribution.sync_owner_profile_if_authenticated",
                    side_effect=OSError("gh unavailable"))):
            result = record_probe_evidence(native_report(), project_root=self.root)
        self.assertTrue(Path(result["local_evidence_file"]).is_file())
        self.assertIsNotNone(result["local_profile_update"])
        self.assertEqual(result["status"], "native_probe_recording_or_sync_failed")
        self.assertFalse(result["runtime_profile_published"])

    def test_no_profiling_for_unauthorized_or_worker_failure(self):
        for status in ("gpu_load_not_authorized",
                       "gpu_load_worker_failed_or_no_result"):
            with self.subTest(status=status):
                with patch("ruth_runtime.profile_registry.record_local_profile",
                           side_effect=AssertionError("Must not record")):
                    result = record_probe_evidence(
                        native_report(status=status), project_root=self.root)
                    self.assertEqual(result["status"], "native_probe_not_recorded")

    def test_duplicate_probe_evidence_stays_single_immutable_file(self):
        with (patch("ruth_runtime.profile_registry.record_local_profile",
                    return_value={
                        "status": "local_profile_already_recorded",
                        "profile_file": str(self.profile),
                    }),
              patch("ruth_runtime.profile_contribution.build_proposal",
                    return_value={"proposal": {"safe": True}}),
              patch("ruth_runtime.profile_contribution.sync_owner_profile_if_authenticated",
                    return_value={"status": "git_profile_already_current",
                                  "git_updated": False})):
            a = record_probe_evidence(native_report(), project_root=self.root)
            b = record_probe_evidence(native_report(), project_root=self.root)
        self.assertEqual(a["local_evidence_file"], b["local_evidence_file"])
        self.assertEqual(len(list((self.root / ".ruth-build/profiles/evidence").glob("*.json"))), 1)
        self.assertTrue(b["runtime_profile_published"])

    def test_cli_auto_records_after_probe_and_reports_publication(self):
        fake = native_report()
        recording = {
            "status": "local_profile_updated",
            "local_profile_update": {"status": "local_profile_updated"},
            "git_publication": {
                "status": "git_profile_updated_model_profiling",
                "git_updated": True},
            "runtime_profile_published": True,
            "local_evidence_file": "/fake/local/evidence.json",
        }
        with (patch("ruth_runtime.load_only_probe.probe_model_load",
                    return_value=fake),
              patch("ruth_runtime.probe_profile_sync.record_probe_evidence",
                    return_value=recording) as recorder,
              patch("sys.stdout", new_callable=StringIO) as out):
            status = main([
                "models", "probe-load", "--quantization", "Q4_K_M",
                "--load-mode", "context", "--allow-gpu-load"])
        self.assertEqual(status, 0)
        self.assertTrue(recorder.call_args.kwargs["sync_git"])
        output = json.loads(out.getvalue())
        self.assertEqual(output["git_publication"]["status"],
                         "git_profile_updated_model_profiling")
        self.assertTrue(output["runtime_profile_published"])

    def test_cli_git_opt_out_keeps_auto_local_recording(self):
        with (patch("ruth_runtime.load_only_probe.probe_model_load",
                    return_value=native_report()),
              patch("ruth_runtime.probe_profile_sync.record_probe_evidence",
                    return_value={
                        "status": "local_profile_updated",
                        "local_profile_update": {"status": "local_profile_updated"},
                        "git_publication": {
                            "status": "git_sync_disabled", "git_updated": False},
                        "runtime_profile_published": False,
                    }) as recorder,
              patch("sys.stdout", new_callable=StringIO) as out):
            code = main([
                "models", "probe-load", "--quantization", "Q4_K_M",
                "--allow-gpu-load", "--no-git-sync"])
        self.assertEqual(code, 0)
        self.assertFalse(recorder.call_args.kwargs["sync_git"])
        self.assertFalse(json.loads(out.getvalue())["runtime_profile_published"])

    def test_cli_unauthorized_probe_never_records(self):
        with (patch("ruth_runtime.load_only_probe.probe_model_load",
                    return_value={"status": "gpu_load_not_authorized"}),
              patch("ruth_runtime.probe_profile_sync.record_probe_evidence",
                    side_effect=AssertionError("Never record"))):
            self.assertEqual(main(["models", "probe-load"]), 2)


if __name__ == "__main__":
    unittest.main()
