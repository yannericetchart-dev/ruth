"""Launcher regression: no host pip, hidden downloads or fake LLM readiness."""
from dataclasses import replace
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ruth_runtime.launcher import build_launch_plan, main
from ruth_runtime.venv_packages import inspect_python_environment


def environment():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        venv = root / ".venv"
        venv.mkdir()
        (venv / "pyvenv.cfg").write_text(
            "home=/usr/bin\ninclude-system-site-packages=false\n"
        )
        def absent(name):
            from importlib.metadata import PackageNotFoundError
            raise PackageNotFoundError(name)
        return inspect_python_environment(
            project_root=root, prefix=venv, base_prefix=Path("/usr"),
            python_version=(3, 12, 3), package_version=absent,
        )


class LauncherTests(unittest.TestCase):
    def test_plan_blocks_outside_project_venv_without_host_probing(self):
        invalid = replace(environment(), interpreter_in_project_venv=False,
                          venv_status="invalid_or_inactive")
        called = []
        plan = build_launch_plan(
            environment=invalid, source_plan_factory=lambda: called.append(1))
        self.assertEqual(called, [])
        self.assertEqual(plan["next_gate"], "activate-or-create-project-venv")
        self.assertFalse(plan["runtime_ready"])
        self.assertFalse(plan["cpu_fallback"])

    def test_source_status_only_affects_setup_not_runtime_readiness(self):
        from types import SimpleNamespace
        plan = build_launch_plan(
            environment=environment(),
            source_plan_factory=lambda: SimpleNamespace(
                build_prerequisites_met=True, source_verified=True,
                to_dict=lambda: {"source_verified": True}
            ))
        self.assertTrue(plan["setup_ready"])
        self.assertFalse(plan["install_ready"])
        self.assertFalse(plan["runtime_ready"])
        self.assertEqual(plan["next_gate"], "complete-hashed-python-dependency-lock")

    def test_no_auto_download_without_setup_approval(self):
        from types import SimpleNamespace
        p = {"setup_ready": True, "next_gate": "acquire-pinned-sources"}
        with patch("ruth_runtime.launcher.inspect_python_environment",
                   return_value=environment()), \
             patch("ruth_runtime.launcher.build_launch_plan", return_value=p), \
             patch("sys.stdout", new_callable=StringIO) as out, \
             patch("ruth_runtime.source_acquisition.acquire_pinned_sources",
                   side_effect=AssertionError("no download")):
            self.assertEqual(main(["setup"]), 2)
            self.assertIn('"download_not_authorized"', out.getvalue())

    def test_setup_approved_still_stops_at_dependency_lock(self):
        from types import SimpleNamespace
        p = {"setup_ready": True, "runtime_ready": False}
        source = SimpleNamespace(source_verified=True, to_dict=lambda: {"status": "verified"})
        with patch("ruth_runtime.launcher.inspect_python_environment",
                   return_value=environment()), \
             patch("ruth_runtime.launcher.build_launch_plan",
                   side_effect=[p, p]), \
             patch("ruth_runtime.source_build_plan.create_source_build_plan",
                   return_value="pinned plan"), \
             patch("ruth_runtime.source_acquisition.acquire_pinned_sources",
                   return_value=source) as acquire, \
             patch("sys.stdout", new_callable=StringIO) as out:
            self.assertEqual(main(["setup", "--allow-downloads"]), 0)
            acquire.assert_called_once_with("pinned plan")
            self.assertIn("sources_verified_dependency_lock_pending", out.getvalue())
            self.assertIn('"pip_started": false', out.getvalue())

    def test_dependency_download_approval_only_stages_candidate(self):
        from types import SimpleNamespace
        report = {"setup_ready": True, "runtime_ready": False}
        source = SimpleNamespace(
            source_verified=True,
            to_dict=lambda: {"source_verified": True},
        )
        candidate = SimpleNamespace(
            status="candidate_review_required",
            to_dict=lambda: {"status": "candidate_review_required",
                             "installation_started": False},
        )
        with patch("ruth_runtime.launcher.inspect_python_environment",
                   return_value=environment()), \
             patch("ruth_runtime.launcher.build_launch_plan",
                   return_value=report), \
             patch("ruth_runtime.source_build_plan.create_source_build_plan",
                   return_value=source), \
             patch("ruth_runtime.dependency_resolution.stage_dependency_candidate",
                   return_value=candidate) as resolve, \
             patch("ruth_runtime.source_acquisition.acquire_pinned_sources",
                   side_effect=AssertionError("existing sources must not re-download")), \
             patch("sys.stdout", new_callable=StringIO) as out:
            code = main(["setup", "--allow-python-dependency-downloads"])
            self.assertEqual(code, 0)
            self.assertEqual(resolve.call_count, 1)
            self.assertTrue(resolve.call_args.kwargs["authorize_network"])
            self.assertIn('"installation_started": false', out.getvalue())
            self.assertIn('"build_started": false', out.getvalue())

    def test_setup_without_any_download_permission_never_resolves(self):
        with patch("ruth_runtime.launcher.inspect_python_environment",
                   return_value=environment()), \
             patch("ruth_runtime.launcher.build_launch_plan",
                   return_value={"setup_ready": True}), \
             patch("ruth_runtime.dependency_resolution.stage_dependency_candidate",
                   side_effect=AssertionError("must not download")), \
             patch("sys.stdout", new_callable=StringIO):
            self.assertEqual(main(["setup"]), 2)

    def test_verify_candidate_runs_metadata_and_offline_resolver_only(self):
        from types import SimpleNamespace
        report = {"setup_ready": True}
        source = SimpleNamespace(
            source_verified=True, to_dict=lambda: {"source_verified": True},
        )
        audit = SimpleNamespace(
            status="security_review_required",
            to_dict=lambda: {"status": "security_review_required",
                             "installation_started": False,
                             "security_review_passed": False},
        )
        with patch("ruth_runtime.launcher.inspect_python_environment",
                   return_value=environment()), \
             patch("ruth_runtime.launcher.build_launch_plan",
                   return_value=report), \
             patch("ruth_runtime.source_build_plan.create_source_build_plan",
                   return_value=source), \
             patch("ruth_runtime.dependency_audit.audit_candidate",
                   return_value=audit) as verifier, \
             patch("ruth_runtime.source_acquisition.acquire_pinned_sources",
                   side_effect=AssertionError("No Git network")), \
             patch("ruth_runtime.dependency_resolution.stage_dependency_candidate",
                   side_effect=AssertionError("No wheel downloads")), \
             patch("sys.stdout", new_callable=StringIO) as output:
            self.assertEqual(main(["setup", "--verify-candidate"]), 0)
            self.assertTrue(verifier.call_args.kwargs["network_authorized"])
            self.assertIn('"security_review_required"', output.getvalue())
            self.assertIn('"installation_started": false', output.getvalue())

    def test_candidate_verification_requires_existing_sources(self):
        from types import SimpleNamespace
        source = SimpleNamespace(
            source_verified=False, to_dict=lambda: {"source_verified": False},
        )
        with patch("ruth_runtime.launcher.inspect_python_environment",
                   return_value=environment()), \
             patch("ruth_runtime.launcher.build_launch_plan",
                   return_value={"setup_ready": True}), \
             patch("ruth_runtime.source_build_plan.create_source_build_plan",
                   return_value=source), \
             patch("ruth_runtime.dependency_audit.audit_candidate",
                   side_effect=AssertionError("No audit without source")), \
             patch("sys.stdout", new_callable=StringIO) as output:
            self.assertEqual(main(["setup", "--verify-candidate"]), 2)
            self.assertIn("source_checkout_unverified", output.getvalue())

    def test_run_never_pretends_llm_engine_implemented(self):
        with patch("ruth_runtime.launcher.inspect_python_environment",
                   return_value=environment()), \
             patch("ruth_runtime.launcher.build_launch_plan", return_value={}), \
             patch("sys.stdout", new_callable=StringIO) as out:
            self.assertEqual(main(["run"]), 2)
            self.assertIn("blocked_unverified_cuda_runtime", out.getvalue())

    def test_doctor_and_plan_do_not_install(self):
        env = environment()
        with patch("ruth_runtime.launcher.inspect_python_environment", return_value=env), \
             patch("ruth_runtime.launcher.build_launch_plan",
                   return_value={"runtime_ready": False}), \
             patch("sys.stdout", new_callable=StringIO) as out:
            self.assertEqual(main(["doctor"]), 0)
            self.assertEqual(main(["plan"]), 0)
            self.assertIn('"action": "doctor"', out.getvalue())
            self.assertIn('"action": "plan"', out.getvalue())

    def test_shell_launcher_exists_and_targets_project_venv(self):
        root = Path(__file__).resolve().parents[1]
        shell = (root / "ruth").read_text()
        self.assertIn('VENV="$ROOT/.venv"', shell)
        self.assertIn('exec "$VENV/bin/python"', shell)
        self.assertNotIn("sudo ", shell)


if __name__ == "__main__":
    unittest.main()
