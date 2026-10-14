"""Offline installation-plan tests: no wheel downloads or installations."""
import unittest

from ruth_runtime.dependencies import DependencyCheck, DependencyReport
from ruth_runtime.install_plan import (
    WHEEL_FILENAME, WHEEL_SHA256, WHEEL_SIZE_BYTES, WHEEL_SOURCE,
    build_installation_plan, parse_glibc_version,
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


class PlanTests(unittest.TestCase):
    def plan(self, checks=None, **kwargs):
        return build_installation_plan(
            dependencies=checks or dependencies(),
            libc=kwargs.pop("libc", ("glibc", "2.39")),
            free_bytes=kwargs.pop("free_bytes", 6_000_000_000),
            interpreter="/repo/.venv/bin/python",
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

    def test_json_compatible_payload_and_no_background_actions(self):
        import json
        p = self.plan()
        exported = p.to_dict()
        self.assertEqual(exported["installation_authorized"], False)
        self.assertEqual(exported["installation_implemented"], False)
        self.assertEqual(json.loads(json.dumps(exported))["version"], "0.3.36")


if __name__ == "__main__":
    unittest.main()
