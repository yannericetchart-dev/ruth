"""Offline installation-plan tests: no wheel downloads or installations."""
import unittest

from ruth_runtime.dependencies import DependencyCheck, DependencyReport
from ruth_runtime.toolkit import ToolkitCheck
from ruth_runtime.install_plan import (
    WHEEL_FILENAME, WHEEL_SHA256, WHEEL_SIZE_BYTES, WHEEL_SOURCE,
    build_installation_plan, parse_glibc_version,
    PRIMARY_METHOD, FALLBACK_METHOD, SOURCE_COMMIT, LLAMA_CPP_SUBMODULE_COMMIT,
    DIRECT_DEPENDENCIES, TRANSITIVE_DEPENDENCIES, DISKCACHE_ADVISORY,
)


REQUIRED = (
    "GNU/Linux x86_64",
    "Python >=3.10 (Ruth .venv)",
    "Ruth .venv",
    "nvidia-smi",
    "GPU/driver NVIDIA",
    "CUDA supportata dal driver",
)


def dependencies(**overrides):
    checks = tuple(
        DependencyCheck(name, True, overrides.get(name, "ok"), "fixture")
        for name in REQUIRED
    ) + (DependencyCheck("llama-cpp-python", True, "missing", "not installed"),)
    return DependencyReport(checks)


def toolkit(*, status="missing", native_ready=False):
    return ToolkitCheck(
        status=status, expected_version="13.2",
        detected_version="13.2" if native_ready else None,
        nvcc_path="/usr/local/cuda-13.2/bin/nvcc" if native_ready else None,
        cmake_path="/usr/bin/cmake", cpp_compiler_path="/usr/bin/c++",
        cuda_architecture="89", native_build_prerequisites=native_ready,
        detail="fixture", remediation="install Toolkit" if not native_ready else None,
    )


class PlanTests(unittest.TestCase):
    def plan(self, checks=None, **kwargs):
        return build_installation_plan(
            dependencies=checks or dependencies(),
            libc=kwargs.pop("libc", ("glibc", "2.39")),
            free_bytes=kwargs.pop("free_bytes", 6_000_000_000),
            interpreter="/repo/.venv/bin/python",
            toolkit=kwargs.pop("toolkit", toolkit()),
            **kwargs,
        )

    def test_valid_machine_may_review_but_cannot_install(self):
        p = self.plan()
        self.assertTrue(p.eligible_for_review)
        self.assertFalse(p.installation_authorized)
        self.assertFalse(p.installation_implemented)
        self.assertEqual(p.wheel_download_fits, True)
        self.assertEqual(p.version, "0.3.36")
        self.assertEqual(p.cuda_variant, "cu132")

    def test_pinned_upstream_artifact_identity(self):
        p = self.plan()
        self.assertEqual(p.wheel_filename, WHEEL_FILENAME)
        self.assertEqual(p.wheel_url, WHEEL_SOURCE)
        self.assertEqual(p.sha256, WHEEL_SHA256)
        self.assertEqual(len(p.sha256), 64)
        self.assertEqual(p.download_bytes, WHEEL_SIZE_BYTES)
        self.assertIn("v0.3.36-cu132", p.wheel_url)
        self.assertIn("manylinux_2_35_x86_64", p.wheel_filename)

    def test_glibc_version_rejected_when_too_old(self):
        self.assertEqual(parse_glibc_version("2.35"), (2, 35))
        self.assertEqual(parse_glibc_version("2.35.1"), (2, 35))
        self.assertIsNone(parse_glibc_version("garbage"))
        self.assertFalse(self.plan(libc=("glibc", "2.34")).eligible_for_review)
        self.assertFalse(self.plan(libc=("musl", "2.39")).eligible_for_review)
        self.assertTrue(self.plan(libc=("glibc", "2.35")).eligible_for_review)

    def test_missing_driver_or_cuda_blocks_wheel_review(self):
        for name in ("GPU/driver NVIDIA", "CUDA supportata dal driver", "Ruth .venv"):
            with self.subTest(name=name):
                p = self.plan(dependencies(**{name: "missing"}))
                self.assertFalse(p.eligible_for_review)
                self.assertEqual(p.runtime_preflight, "blocked")

    def test_missing_target_package_expected_during_planning(self):
        p = self.plan()
        self.assertEqual(p.runtime_preflight, "ok")
        self.assertIn("numpy", p.unpinned_dependencies)
        self.assertIn("jinja2", p.unpinned_dependencies)
        self.assertFalse(p.installation_implemented)

    def test_no_free_space_is_not_success(self):
        p = self.plan(free_bytes=WHEEL_SIZE_BYTES - 1)
        self.assertFalse(p.eligible_for_review)
        self.assertFalse(p.wheel_download_fits)
        empty = self.plan(free_bytes=None, disk_usage=lambda _: (_ for _ in ()).throw(OSError("blocked")))
        self.assertFalse(empty.eligible_for_review)
        self.assertIsNone(empty.wheel_download_fits)

    def test_transitive_dependency_closure_is_explicit(self):
        p = self.plan()
        self.assertEqual(len(DIRECT_DEPENDENCIES), 4)
        self.assertIn("jinja2>=2.11.3", p.direct_requirements)
        self.assertIn("MarkupSafe>=2.0", TRANSITIVE_DEPENDENCIES)
        self.assertIn("markupsafe", p.unpinned_dependencies)
        self.assertEqual(len(p.unpinned_dependencies), 5)
        self.assertFalse(p.dependency_lock_ready)

    def test_diskcache_security_warning_cannot_be_ignored(self):
        p = self.plan()
        self.assertTrue(any(DISKCACHE_ADVISORY in warning
                            for warning in p.security_advisories))
        self.assertIn("DISABLED", p.disk_cache_policy)
        self.assertIn("LlamaDiskCache", p.disk_cache_policy)
        self.assertFalse(p.installation_ready)
        self.assertTrue(p.eligible_for_review)
        self.assertFalse(p.installation_authorized)

    def test_no_machine_qualifies_for_installation_yet(self):
        p = self.plan()
        self.assertFalse(p.installation_ready)
        self.assertFalse(p.installation_implemented)
        self.assertFalse(p.installation_authorized)

    def test_native_build_takes_priority_over_wheel(self):
        p = self.plan(toolkit=toolkit(status="ok", native_ready=True))
        self.assertEqual(p.strategy_priority, (PRIMARY_METHOD, FALLBACK_METHOD))
        self.assertEqual(p.primary_method, "cuda-toolkit-native-build")
        self.assertEqual(p.primary_toolkit_package, "cuda-toolkit-13-2")
        self.assertTrue(p.primary_prerequisites_met)
        self.assertEqual(p.native_build_cmake_args,
                         "-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89")
        self.assertEqual(p.cuda_architecture, "89")
        self.assertEqual(len(SOURCE_COMMIT), 40)
        self.assertEqual(len(LLAMA_CPP_SUBMODULE_COMMIT), 40)
        self.assertIn("scikit-build-core", p.source_build_requirements[0])
        self.assertFalse(p.installation_ready)

    def test_no_toolkit_requires_remediation_not_autofallback(self):
        p = self.plan(toolkit=toolkit(status="missing"))
        self.assertFalse(p.primary_prerequisites_met)
        self.assertTrue(p.fallback_review_eligible)
        self.assertFalse(p.fallback_automatically_selected)
        self.assertTrue(p.fallback_requires_explicit_authorization)
        self.assertFalse(p.installation_authorized)
        self.assertEqual(p.toolkit["status"], "missing")

    def test_invalid_toolkit_must_not_promote_source(self):
        p = self.plan(toolkit=toolkit(status="incompatible"))
        self.assertFalse(p.primary_prerequisites_met)
        self.assertEqual(p.strategy_priority[0], PRIMARY_METHOD)
        self.assertFalse(p.fallback_automatically_selected)

    def test_source_review_possible_when_wheel_not_compatible(self):
        p = self.plan(toolkit=toolkit(status="ok", native_ready=True),
                      libc=("glibc", "2.34"), free_bytes=12)
        self.assertTrue(p.eligible_for_review)
        self.assertTrue(p.primary_prerequisites_met)
        self.assertFalse(p.fallback_review_eligible)
        self.assertFalse(p.installation_ready)

    def test_json_compatible_payload_and_no_background_actions(self):
        import json
        p = self.plan()
        exported = p.to_dict()
        self.assertEqual(exported["installation_authorized"], False)
        self.assertEqual(exported["installation_implemented"], False)
        self.assertEqual(json.loads(json.dumps(exported))["version"], "0.3.36")


if __name__ == "__main__":
    unittest.main()
