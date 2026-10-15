"""18-case Gemma3 matrix runner: offline mocks, no GPU and no inference."""
from __future__ import annotations

from io import StringIO
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from ruth_runtime.gemma3_context_matrix import (
    MATRIX, command_for_case, execute_matrix, main, matrix_cases,
)


def successful_native_report(row):
    k, v = row["type_k"], row["type_v"]
    return {
        "candidate": "gemma-3-1b-it/gguf/BF16",
        "status": "gpu_load_only_completed",
        "mode": "context",
        "context_initialized": True,
        "full_gpu_offload_observed": True,
        "gpu_process_observed": True,
        "cpu_fallback_selected": False,
        "inference_executed": False,
        "generated_tokens": 0,
        "context_options_requested": {
            "flash_attention": row["flash_attention"],
            "type_k": k, "type_v": v,
        },
        "profile_recording_status": "local_profile_updated",
        "local_evidence_file": "/local/evidence/pinned.json",
        "git_publication": {"status": "git_sync_disabled",
                            "git_updated": False},
        "native_kv_log": {
            "flash_attention_native_parameter": (
                "enabled" if row["flash_attention"] == "on" else "disabled"),
            "n_ctx_effective_native": 256,
            "n_ctx_seq_effective_native": 256,
            "n_batch_effective_native": 64,
            "n_ubatch_effective_native": 64,
            "n_seq_max_native": 1,
            "global_kv": {
                "cells": 256,
                "native_logical_kv": {
                    "layers": 4, "key_type": k.lower(),
                    "value_type": v.lower(),
                },
            },
            "swa_kv": {
                "cells": 256,
                "native_logical_kv": {
                    "layers": 22, "key_type": k.lower(),
                    "value_type": v.lower(),
                },
            },
        },
        "native_buffers": [
            {"kind": "kv", "device": "CUDA0",
             "mib": row["kv_logical_mib_source_derived"]},
            {"kind": "compute", "device": "CUDA0", "mib": 64.28},
        ],
    }


class MatrixPlanningTests(unittest.TestCase):
    def test_plan_is_exactly_eighteen_and_prior_baselines_are_first(self):
        rows = matrix_cases()
        self.assertEqual(len(rows), 18)
        self.assertEqual(len(MATRIX), 18)
        self.assertEqual([r["case"] for r in rows], list(range(1,19)))
        self.assertEqual([r["flash_attention"] for r in rows[:2]], ["off","on"])
        self.assertEqual([(r["type_k"],r["type_v"]) for r in rows[:2]],
                         [("F16","F16"),("BF16","BF16")])
        self.assertEqual(len({
            (r["flash_attention"],r["type_k"],r["type_v"]) for r in rows
        }), 18)
        self.assertEqual(sorted(set(r["kv_logical_mib_source_derived"] for r in rows)),
                         [6.5, 9.75, 13.0])
        self.assertTrue(all(r["inference_approved"] is False for r in rows))

    def test_static_commands_have_exact_pins_and_no_inference_or_git_sync(self):
        root = Path("/tmp/test-ruth-project")
        for row in matrix_cases():
            command = command_for_case(row, root)
            self.assertEqual(command[:3],
                             [str(root / ".venv/bin/python"), "-m", "ruth_runtime"])
            self.assertIn("--allow-gpu-load", command)
            self.assertIn("--no-git-sync", command)
            self.assertNotIn("--allow-gpu-inference", command)
            self.assertNotIn("--cuda-preinit", command)
            self.assertEqual(command[command.index("--load-mode")+1], "context")
            self.assertEqual(command[command.index("--type-k")+1], row["type_k"])
            self.assertEqual(command[command.index("--type-v")+1], row["type_v"])
            self.assertEqual(command[command.index("--flash-attn")+1],
                             row["flash_attention"])

    def test_read_only_plan_never_spawns_worker(self):
        with patch("sys.stdout", new_callable=StringIO) as stdout, \
             patch("ruth_runtime.gemma3_context_matrix.execute_matrix",
                   side_effect=AssertionError("No CUDA or child subprocess")):
            self.assertEqual(main([]), 0)
            plan=json.loads(stdout.getvalue())
        self.assertEqual(len(plan["cases"]), 18)
        self.assertFalse(plan["gpu_allocations_executed"])
        with patch("sys.stdout", new_callable=StringIO) as stdout:
            self.assertEqual(main(["--case","3"]),0)
            self.assertEqual(len(json.loads(stdout.getvalue())["cases"]),1)
        with patch("sys.stdout", new_callable=StringIO) as stdout:
            self.assertEqual(main(["--start-at","3"]),0)
            self.assertEqual(len(json.loads(stdout.getvalue())["cases"]),16)

    def test_explicit_allocation_authorization_is_mandatory(self):
        with patch("sys.stderr", new_callable=StringIO):
            with self.assertRaises(SystemExit) as exc:
                main(["--execute"])
            self.assertEqual(exc.exception.code,2)
            with self.assertRaises(SystemExit) as exc:
                main(["--allow-gpu-load"])
            self.assertEqual(exc.exception.code,2)
            with self.assertRaises(SystemExit) as exc:
                main(["--execute","--allow-gpu-load","--start-at","4",
                      "--case","3"])
            self.assertEqual(exc.exception.code,2)


class MatrixExecutionTests(unittest.TestCase):
    def fixture(self):
        tmp=TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root=Path(tmp.name)
        interpreter=root / ".venv/bin/python"
        interpreter.parent.mkdir(parents=True)
        interpreter.write_text("")
        return root,interpreter

    def test_sequential_run_verifies_native_dtype_records_for_all_cases(self):
        root,interpreter=self.fixture()
        calls=[]
        cases=matrix_cases()
        def fake_run(argv,**kwargs):
            calls.append((argv,kwargs))
            k=argv[argv.index("--type-k")+1]
            v=argv[argv.index("--type-v")+1]
            flash=argv[argv.index("--flash-attn")+1]
            row=next(c for c in cases if
                     (c["flash_attention"],c["type_k"],c["type_v"])
                     == (flash,k,v))
            return subprocess.CompletedProcess(
                argv,0,json.dumps(successful_native_report(row)),"")
        with patch("ruth_runtime.gemma3_context_matrix.sys.executable",
                   str(interpreter)), \
             patch("sys.stderr", new_callable=StringIO):
            report=execute_matrix(cases=cases,project_root=root,run=fake_run)
        self.assertEqual(report["status"],"selected_allocations_completed")
        self.assertEqual(report["cases_run"],18)
        self.assertEqual(report["context_allocations_observed"],18)
        self.assertEqual(len(calls),18)
        self.assertEqual(report["stopped_at"],None)
        self.assertTrue(all(item["native_kv_types_verified"]
                            for item in report["results"]))
        self.assertTrue(all(item["status"]=="native_context_verified"
                            for item in report["results"]))
        for argv,opts in calls:
            self.assertEqual(argv[0],str(root / ".venv/bin/python"))
            self.assertEqual(opts["cwd"],str(root))
            self.assertNotIn("--allow-gpu-inference",argv)
            self.assertIn("--no-git-sync",argv)
        self.assertFalse(report["production_approved"])
        self.assertFalse(report["inference_executed"])

    def test_stop_after_first_error_without_fallback(self):
        root,interpreter=self.fixture()
        calls=[]
        rows=matrix_cases()[2:5]
        def fake_run(argv,**kw):
            calls.append(argv)
            row=rows[len(calls)-1]
            outcome=successful_native_report(row)
            if len(calls)==2:
                outcome["status"]="gpu_load_only_failed"
                outcome["context_initialized"]=False
            return subprocess.CompletedProcess(argv, 0 if len(calls)==1 else 2,
                                               json.dumps(outcome),"")
        with patch("ruth_runtime.gemma3_context_matrix.sys.executable",
                   str(interpreter)), \
             patch("sys.stderr", new_callable=StringIO):
            result=execute_matrix(cases=rows,project_root=root,run=fake_run)
        self.assertEqual(result["status"],"stopped_after_unverified_or_failed_case")
        self.assertEqual(result["stopped_at"],rows[1]["case"])
        self.assertEqual(result["cases_run"],2)
        self.assertEqual(len(calls),2)
        self.assertFalse(result["inference_executed"])

    def test_wrong_native_cache_dtype_prevents_success(self):
        root,interpreter=self.fixture()
        row=matrix_cases()[2]
        payload=successful_native_report(row)
        payload["native_kv_log"]["swa_kv"]["native_logical_kv"]["key_type"]="f16"
        with patch("ruth_runtime.gemma3_context_matrix.sys.executable",
                   str(interpreter)), \
             patch("sys.stderr", new_callable=StringIO):
            outcome=execute_matrix(
                cases=[row],project_root=root,
                run=lambda args,**kw: subprocess.CompletedProcess(
                    args,0,json.dumps(payload),""))
        self.assertFalse(outcome["results"][0]["native_kv_types_verified"])
        self.assertEqual(outcome["stopped_at"],row["case"])


if __name__ == "__main__":
    unittest.main()
