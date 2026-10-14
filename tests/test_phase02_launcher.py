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

    def test_security_review_runs_offline_and_never_installs(self):
        from types import SimpleNamespace
        plan = SimpleNamespace(source_verified=True, to_dict=lambda: {})
        gate = SimpleNamespace(
            mitigations_validated=True,
            status="controls_verified_residual_risk",
            to_dict=lambda: {
                "lock_approved": False,
                "security_review_passed": False,
                "mitigations_validated": True,
            },
        )
        with patch("ruth_runtime.launcher.inspect_python_environment",
                   return_value=environment()), \
             patch("ruth_runtime.launcher.build_launch_plan",
                   return_value={"setup_ready": True}), \
             patch("ruth_runtime.source_build_plan.create_source_build_plan",
                   return_value=plan), \
             patch("ruth_runtime.security_gate.review_diskcache_risk",
                   return_value=gate) as review, \
             patch("ruth_runtime.dependency_resolution.stage_dependency_candidate",
                   side_effect=AssertionError("No downloads")), \
             patch("ruth_runtime.source_acquisition.acquire_pinned_sources",
                   side_effect=AssertionError("No cloning")), \
             patch("sys.stdout", new_callable=StringIO) as output:
            self.assertEqual(main(["setup", "--review-security"]), 0)
            self.assertEqual(review.call_count, 1)
            self.assertIn('"lock_approved": false', output.getvalue())
            self.assertIn('"installation_started": false', output.getvalue())
            self.assertIn("controls_verified_residual_risk", output.getvalue())

    def test_security_review_failure_blocks(self):
        from types import SimpleNamespace
        gate = SimpleNamespace(
            mitigations_validated=False, status="blocked",
            to_dict=lambda: {"errors": ["Pinned cache source changed"]},
        )
        with patch("ruth_runtime.launcher.inspect_python_environment",
                   return_value=environment()), \
             patch("ruth_runtime.launcher.build_launch_plan",
                   return_value={"setup_ready": True}), \
             patch("ruth_runtime.source_build_plan.create_source_build_plan",
                   return_value=SimpleNamespace(source_verified=True)), \
             patch("ruth_runtime.security_gate.review_diskcache_risk",
                   return_value=gate), \
             patch("sys.stdout", new_callable=StringIO) as output:
            self.assertEqual(main(["setup", "--review-security"]), 2)
            self.assertIn('"status": "blocked"', output.getvalue())

    def test_explicit_native_build_stages_only_wheel(self):
        from types import SimpleNamespace
        plan = SimpleNamespace(source_verified=True, to_dict=lambda: {})
        result = SimpleNamespace(
            status="native_cuda_wheel_built",
            to_dict=lambda: {
                "native_build_started": True,
                "installed_in_ruth_venv": False,
                "fallback_automatically_selected": False,
            },
        )
        with patch("ruth_runtime.launcher.inspect_python_environment",
                   return_value=environment()), \
             patch("ruth_runtime.launcher.build_launch_plan",
                   return_value={"setup_ready": True}), \
             patch("ruth_runtime.source_build_plan.create_source_build_plan",
                   return_value=plan), \
             patch("ruth_runtime.native_builder.build_native_cuda_wheel",
                   return_value=result) as builder, \
             patch("ruth_runtime.source_acquisition.acquire_pinned_sources",
                   side_effect=AssertionError("No Git download")), \
             patch("ruth_runtime.dependency_resolution.stage_dependency_candidate",
                   side_effect=AssertionError("No PyPI download")), \
             patch("sys.stdout", new_callable=StringIO) as output:
            self.assertEqual(main(["setup", "--build-native"]), 0)
            self.assertEqual(builder.call_count, 1)
            self.assertIn('"installation_started": false', output.getvalue())
            self.assertIn('"risk_todo_open": true', output.getvalue())

    def test_native_build_never_auto_fallback_on_failure(self):
        from types import SimpleNamespace
        failure = SimpleNamespace(
            status="cuda_compile_failed",
            to_dict=lambda: {"fallback_automatically_selected": False},
        )
        with patch("ruth_runtime.launcher.inspect_python_environment",
                   return_value=environment()), \
             patch("ruth_runtime.launcher.build_launch_plan",
                   return_value={"setup_ready": True}), \
             patch("ruth_runtime.source_build_plan.create_source_build_plan",
                   return_value=SimpleNamespace(source_verified=True)), \
             patch("ruth_runtime.native_builder.build_native_cuda_wheel",
                   return_value=failure), \
             patch("sys.stdout", new_callable=StringIO) as output:
            self.assertEqual(main(["setup", "--build-native"]), 2)
            self.assertIn('"fallback_automatically_selected": false', output.getvalue())

    def test_native_build_conflicting_network_options_denied(self):
        from types import SimpleNamespace
        with patch("ruth_runtime.launcher.inspect_python_environment",
                   return_value=environment()), \
             patch("ruth_runtime.launcher.build_launch_plan",
                   return_value={"setup_ready": True}), \
             patch("ruth_runtime.source_build_plan.create_source_build_plan",
                   return_value=SimpleNamespace(source_verified=True)), \
             patch("ruth_runtime.native_builder.build_native_cuda_wheel",
                   side_effect=AssertionError("Never build with ambiguous consent")), \
             patch("sys.stdout", new_callable=StringIO) as output:
            self.assertEqual(main([
                "setup", "--build-native", "--allow-downloads"
            ]), 2)
            self.assertIn('"conflicting_build_options"', output.getvalue())

    def test_native_build_conflicts_rejected_before_environment_or_io(self):
        forbidden_flags = (
            "--allow-downloads", "--allow-python-dependency-downloads",
            "--verify-candidate", "--review-security",
        )
        for flag in forbidden_flags:
            with self.subTest(flag=flag), \
                 patch("ruth_runtime.launcher.inspect_python_environment",
                       side_effect=AssertionError("conflicts must block before venv probe")), \
                 patch("ruth_runtime.launcher.build_launch_plan",
                       side_effect=AssertionError("conflicts must block before preflight")), \
                 patch("ruth_runtime.source_acquisition.acquire_pinned_sources",
                       side_effect=AssertionError("must not clone")), \
                 patch("ruth_runtime.native_builder.build_native_cuda_wheel",
                       side_effect=AssertionError("must not compile")), \
                 patch("sys.stdout", new_callable=StringIO) as output:
                self.assertEqual(main(["setup", "--build-native", flag]), 2)
                self.assertIn('"conflicting_build_options"', output.getvalue())
                self.assertIn('"installation_started": false', output.getvalue())
                self.assertIn('"build_started": false', output.getvalue())

    def test_native_install_preflight_is_read_only(self):
        from types import SimpleNamespace
        outcome = SimpleNamespace(
            status="ready_for_explicit_install_authorization",
            to_dict=lambda: {"native_wheel_verified": True, "installation_started": False},
        )
        with patch("ruth_runtime.launcher.inspect_python_environment",
                   return_value=environment()), \
             patch("ruth_runtime.launcher.build_launch_plan",
                   return_value={"setup_ready": True}), \
             patch("ruth_runtime.source_build_plan.create_source_build_plan",
                   return_value=SimpleNamespace(source_verified=True)), \
             patch("ruth_runtime.native_installer.inspect_native_install_inputs",
                   return_value=(outcome, ())) as inspection, \
             patch("ruth_runtime.native_installer.install_native_cuda",
                   side_effect=AssertionError("No install in plan")), \
             patch("sys.stdout", new_callable=StringIO) as output:
            self.assertEqual(main(["setup", "--plan-native-install"]), 0)
            self.assertEqual(inspection.call_count, 1)
            self.assertIn('"installation_started": false', output.getvalue())

    def test_native_install_passes_separate_explicit_risk_consent(self):
        from types import SimpleNamespace
        outcome = SimpleNamespace(
            status="native_cuda_binding_installed",
            to_dict=lambda: {"installed_in_ruth_venv": True,
                             "diskcache_todo_open": True},
        )
        with patch("ruth_runtime.launcher.inspect_python_environment",
                   return_value=environment()), \
             patch("ruth_runtime.launcher.build_launch_plan",
                   return_value={"setup_ready": True}), \
             patch("ruth_runtime.source_build_plan.create_source_build_plan",
                   return_value=SimpleNamespace(source_verified=True)), \
             patch("ruth_runtime.native_installer.install_native_cuda",
                   return_value=outcome) as install, \
             patch("sys.stdout", new_callable=StringIO) as output:
            self.assertEqual(main([
                "setup", "--install-native", "--accept-diskcache-risk"
            ]), 0)
            self.assertTrue(install.call_args.kwargs["authorize_install"])
            self.assertTrue(install.call_args.kwargs["accept_diskcache_risk"])
            self.assertIn('"diskcache_todo_open": true', output.getvalue())

    def test_install_native_without_risk_acknowledgement_is_not_assumed(self):
        from types import SimpleNamespace
        outcome = SimpleNamespace(
            status="explicit_install_and_risk_consent_required",
            to_dict=lambda: {"installation_started": False},
        )
        with patch("ruth_runtime.launcher.inspect_python_environment",
                   return_value=environment()), \
             patch("ruth_runtime.launcher.build_launch_plan",
                   return_value={"setup_ready": True}), \
             patch("ruth_runtime.source_build_plan.create_source_build_plan",
                   return_value=SimpleNamespace(source_verified=True)), \
             patch("ruth_runtime.native_installer.install_native_cuda",
                   return_value=outcome) as install, \
             patch("sys.stdout", new_callable=StringIO) as output:
            self.assertEqual(main(["setup", "--install-native"]), 2)
            self.assertFalse(install.call_args.kwargs["accept_diskcache_risk"])
            self.assertIn("explicit_install_and_risk_consent_required", output.getvalue())

    def test_risk_flag_alone_never_authorizes_actions(self):
        with patch("ruth_runtime.launcher.inspect_python_environment",
                   side_effect=AssertionError("Conflict must be rejected first")), \
             patch("sys.stdout", new_callable=StringIO) as output:
            self.assertEqual(main(["setup", "--accept-diskcache-risk"]), 2)
            self.assertIn("conflicting_build_options", output.getvalue())

    def test_install_cannot_combine_with_network_build_or_audits(self):
        incompatible = ("--allow-downloads", "--allow-python-dependency-downloads",
                        "--build-native", "--verify-candidate",
                        "--review-security", "--plan-native-install")
        for flag in incompatible:
            with self.subTest(flag=flag), \
                 patch("ruth_runtime.launcher.inspect_python_environment",
                       side_effect=AssertionError("No side effects before consent validation")), \
                 patch("sys.stdout", new_callable=StringIO) as output:
                self.assertEqual(main(["setup", "--install-native", flag]), 2)
                self.assertIn("conflicting_build_options", output.getvalue())

    def test_native_shared_library_verification_is_read_only(self):
        from types import SimpleNamespace
        verified = SimpleNamespace(
            status="native_library_dual_abi_verified",
            to_dict=lambda: {
                "shared_libraries_verified": True,
                "python_binding_same_library": True,
                "inference_executed": False,
            },
        )
        with patch("ruth_runtime.launcher.inspect_python_environment",
                   return_value=environment()), \
             patch("ruth_runtime.launcher.build_launch_plan",
                   return_value={"setup_ready": True}), \
             patch("ruth_runtime.source_build_plan.create_source_build_plan",
                   return_value=SimpleNamespace(source_verified=True)), \
             patch("ruth_runtime.native_library.verify_native_library",
                   return_value=verified) as checker, \
             patch("ruth_runtime.native_builder.build_native_cuda_wheel",
                   side_effect=AssertionError("No recompilation")), \
             patch("ruth_runtime.native_installer.install_native_cuda",
                   side_effect=AssertionError("No installation")), \
             patch("sys.stdout", new_callable=StringIO) as output:
            self.assertEqual(main(["setup", "--verify-native-library"]), 0)
            self.assertEqual(checker.call_count, 1)
            self.assertIn('"inference_executed": false', output.getvalue())
            self.assertIn('"installation_started": false', output.getvalue())

    def test_native_shared_library_verification_rejects_combined_actions(self):
        for flag in ("--allow-downloads", "--allow-python-dependency-downloads",
                     "--build-native", "--install-native", "--review-security",
                     "--plan-native-install"):
            with self.subTest(flag=flag), \
                 patch("ruth_runtime.launcher.inspect_python_environment",
                       side_effect=AssertionError("No preflight before flag check")), \
                 patch("sys.stdout", new_callable=StringIO) as output:
                self.assertEqual(main(["setup", "--verify-native-library", flag]), 2)
                self.assertIn('"conflicting_build_options"', output.getvalue())

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
