"""CPU-only tests for phase 2 isolated native mapping verification."""

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from calibration.mapping_phase import (
    native_mapping_trials, probe_native_mapping, run_mapping_phase,
)


def layout(architecture="qwen2", blocks=2):
    tensors = [{"name": "token_embd.weight", "bytes": 300,
                "block": None, "group": "embedding"}]
    for block in range(blocks):
        tensors.extend([
            {"name": f"blk.{block}.attn_q.weight", "bytes": 40,
             "block": block, "group": "attention"},
            {"name": f"blk.{block}.attn_k.weight", "bytes": 20,
             "block": block, "group": "attention"},
            {"name": f"blk.{block}.ffn_down.weight", "bytes": 200,
             "block": block, "group": "feed_forward"},
        ])
    if architecture != "gemma2":
        tensors.append({"name": "output.weight", "bytes": 300,
                        "block": None, "group": "output"})
    return {
        "metadata": {
            "general.architecture": architecture,
            f"{architecture}.block_count": blocks,
        },
        "tensors": tensors,
        "weights_bytes": sum(t["bytes"] for t in tensors),
    }


def native_log(trial, *, broken=None):
    logs = ["llama_model_load_tensors: offloading output layer to GPU"]
    wanted = {
        **{n: "CUDA_Host" for n in trial["host_tensor_names"]},
        **{n: "CUDA0" for n in trial["gpu_tensor_names"]},
    }
    for name in sorted(wanted):
        for _ in range(trial["expected_native_log_occurrences"].get(name, 1)):
            buffer_type = "CUDA0" if broken == name else wanted[name]
            logs.append(
                f"llama_model_loader: tensor {name} (1 MiB Q5) "
                f"buffer type overridden to {buffer_type}"
            )
    return "\n".join(logs)


def native_memory(kv=True):
    return {
        "native_buffer_records": [
            {"category": "kv", "domain": (
                "gpu_vram" if kv else "host_cpu"
            ), "native_buffer": "CUDA0" if kv else "CPU",
             "reported_mib": 2.0},
            {"category": "compute", "domain": "gpu_vram",
             "native_buffer": "CUDA0", "reported_mib": 1.0},
        ],
    }


class NativeMappingPlanningTests(unittest.TestCase):
    def test_all_gguf_tensors_covered_once_for_both_directions(self):
        from calibration.mapping_native_validation import verify_override_logs
        for arch in ("qwen2", "llama", "gemma2"):
            with self.subTest(architecture=arch):
                probes = native_mapping_trials(layout(arch))
                self.assertEqual([p["id"] for p in probes], [
                    "block0_host", "attention_tensor_host",
                ])
                self.assertEqual(probes[0]["host_tensor_count"], 3)
                self.assertEqual(probes[1]["host_tensor_names"],
                                 ["blk.1.attn_k.weight"])
                for probe in probes:
                    self.assertEqual(
                        probe["host_tensor_count"] + probe["gpu_tensor_count"],
                        len(layout(arch)["tensors"]),
                    )
                    self.assertTrue(
                        verify_override_logs(probe, native_log(probe))["verified"]
                    )
                    self.assertTrue(all(rule["pattern"].startswith("^")
                                        for rule in probe["rules"]))
                if arch == "gemma2":
                    self.assertEqual(
                        probes[0]["expected_native_log_occurrences"][
                            "token_embd.weight"
                        ], 2
                    )

    def test_reject_uncovered_or_duplicated_gguf(self):
        example = layout()
        example["tensors"][2]["name"] = example["tensors"][1]["name"]
        with self.assertRaisesRegex(ValueError, "duplicato"):
            native_mapping_trials(example)

    def test_out_of_gpu_log_is_not_falsely_counted_as_cuda_residency(self):
        from calibration.mapping_native_validation import verify_override_logs
        trial = native_mapping_trials(layout())[0]
        host_tensor = trial["host_tensor_names"][0]
        broken = native_log(trial, broken=host_tensor)
        self.assertFalse(
            verify_override_logs(trial, broken)["verified"]
        )


class NativeMappingWorkerVerificationTests(unittest.TestCase):
    def probe(self, *, bad_log=False, kv=True, returncode=0,
              model_id="qwen-2.5-7b-instruct"):
        trial = native_mapping_trials(layout())[0]
        stderr = native_log(
            trial, broken=trial["host_tensor_names"][0] if bad_log else None
        )
        stdout = json.dumps({
            "worker_ok": True, "model_id": model_id,
            "n_ctx": 64, "mode": trial["id"],
            "measurements": {
                "generated_tokens": 2, "vram_peak_sampled_mib": 5300,
                "nvml_errors": [],
            },
        })
        with (
            patch("calibration.mapping_phase.subprocess.run",
                  return_value=SimpleNamespace(
                      returncode=returncode, stdout=stdout, stderr=stderr,
                  )),
            patch("calibration.mapping_phase.parse_native_memory",
                  return_value=native_memory(kv)),
        ):
            outcome = probe_native_mapping(
                Path("/dummy/model.gguf"),
                "qwen-2.5-7b-instruct", trial,
            )
        return outcome

    def test_native_cpu_and_gpu_placements_and_cuda_kv_are_required(self):
        valid = self.probe()
        self.assertTrue(valid["success"])
        self.assertTrue(valid["native_placement"]["verified"])
        self.assertTrue(valid["gpu_runtime_validation"]["verified"])
        self.assertEqual(valid["measurements"]["vram_peak_sampled_mib"], 5300)
        self.assertIsNone(valid["failure_class"])

        bad = self.probe(bad_log=True)
        self.assertFalse(bad["success"])
        self.assertEqual(bad["failure_class"], "tensor_placement_not_verified")

        kv_cpu = self.probe(kv=False)
        self.assertFalse(kv_cpu["success"])
        self.assertEqual(kv_cpu["failure_class"], "gpu_kv_compute_not_verified")

        worker_error = self.probe(returncode=-6)
        self.assertFalse(worker_error["success"])
        self.assertEqual(worker_error["worker_signal"], "SIGABRT")
        self.assertNotEqual(worker_error["failure_class"], "native_cuda_oom")


class MappingPhaseTests(unittest.TestCase):
    def test_pilot_validates_two_probes_without_selecting_profile(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / "qwen.gguf"
            model.write_bytes(b"fake")
            gguf = layout()
            baseline = {
                "model_id": "qwen-2.5-7b-instruct",
                "source": "kaggle", "repository": "qwen-lm/qwen2.5",
                "filename": model.name, "local_path": str(model),
                "quantization": "q5_k_m", "architecture": "qwen2",
                "native_verified_all_gpu": True,
            }
            matrix = {
                "success": True, "status": "enumeration_only",
                "filename": model.name, "model_path": str(model.resolve()),
                "block_count": 2,
                "serialized_weight_bytes": gguf["weights_bytes"],
            }
            ctx = {
                "success": True, "filename": model.name,
                "quantization": "q5_k_m",
                "confirmed_max_gpu_resident_n_ctx": 7424,
            }
            events = []
            def success(path, mid, trial, **kwargs):
                return {
                    "mode": trial["id"], "success": True,
                    "native_placement": {"verified": True},
                    "gpu_runtime_validation": {"verified": True},
                    "measurements": {"vram_peak_sampled_mib": 5300},
                }
            with (
                patch("calibration.mapping_phase.inspect_gguf",
                      return_value=gguf),
                patch("calibration.mapping_phase.probe_native_mapping",
                      side_effect=success) as worker,
            ):
                report = run_mapping_phase(
                    baseline, matrix, ctx, model_id=baseline["model_id"],
                    progress=events.append,
                )
            self.assertTrue(report["success"])
            self.assertEqual(worker.call_count, 2)
            self.assertEqual(report["status"],
                             "diagnostic_native_placement_validated")
            self.assertEqual(report["reference_full_cuda_n_ctx"], 7424)
            self.assertEqual(report["probe_n_ctx"], 64)
            self.assertIsNone(report["selected_mapping"])
            self.assertFalse(report["profile_generated"])
            self.assertFalse(report["runtime_optimized"])
            self.assertEqual(
                len([e for e in events
                     if e.get("event") == "mapping_probe_result"]), 2
            )

    def test_worker_failure_does_not_mark_mapping_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "qwen.gguf"
            path.write_bytes(b"x")
            gguf = layout()
            baseline = {
                "model_id": "qwen-2.5-7b-instruct",
                "filename": path.name, "local_path": str(path),
                "quantization": "q5_k_m", "architecture": "qwen2",
                "native_verified_all_gpu": True,
            }
            matrix = {"success": True, "status": "enumeration_only",
                      "filename": path.name, "model_path": str(path.resolve()),
                      "block_count": 2,
                      "serialized_weight_bytes": gguf["weights_bytes"]}
            context = {"success": True, "filename": path.name,
                       "quantization": "q5_k_m",
                       "confirmed_max_gpu_resident_n_ctx": 7424}
            with (
                patch("calibration.mapping_phase.inspect_gguf",
                      return_value=gguf),
                patch("calibration.mapping_phase.probe_native_mapping",
                      return_value={
                          "mode": "block0_host", "success": False,
                          "failure_class": "worker_protocol_error",
                      }) as worker,
            ):
                result = run_mapping_phase(
                    baseline, matrix, context, model_id=baseline["model_id"]
                )
            self.assertFalse(result["success"])
            self.assertEqual(worker.call_count, 1)
            self.assertEqual(result["status"],
                             "blocked_unverified_native_mapping")


if __name__ == "__main__":
    unittest.main()
