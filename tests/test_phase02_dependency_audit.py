"""Candidate verification test matrix — zero network, pip install or CUDA."""
from dataclasses import asdict
from pathlib import Path
import json
import subprocess
import tempfile
import unittest
from zipfile import ZipFile

from ruth_runtime.dependency_audit import audit_candidate
from ruth_runtime.dependency_resolution import (
    CANDIDATE_NAME, ROOT_REQUIREMENTS, WHEELHOUSE, inspect_wheel,
    normalize_project_name, PYPI_INDEX,
)
from ruth_runtime.install_plan import SOURCE_COMMIT, LLAMA_CPP_SUBMODULE_COMMIT


ARTIFACTS = (
    ("diskcache", "5.6.3"), ("Jinja2", "3.1.6"),
    ("MarkupSafe", "3.0.4"), ("numpy", "2.5.4"),
    ("packaging", "26.3"), ("pathspec", "1.1.1"),
    ("scikit-build-core", "1.1.1"), ("typing-extensions", "4.16.0"),
)


def candidate_fixture(root):
    final = root / WHEELHOUSE / CANDIDATE_NAME
    wheels = final / "wheels"
    wheels.mkdir(parents=True)
    rows = []
    for name, version in ARTIFACTS:
        norm = name.replace("-", "_").lower()
        filename = f"{norm}-{version}-py3-none-any.whl"
        path = wheels / filename
        with ZipFile(path, "w") as zipfile:
            zipfile.writestr(
                f"{norm}-{version}.dist-info/METADATA",
                f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
            )
        rows.append(asdict(inspect_wheel(path)))
    manifest = {
        "schema": "ruth-python-candidate-v1",
        "approved": False, "dependency_lock_ready": False,
        "python": "3.12", "index": PYPI_INDEX,
        "source_commit": SOURCE_COMMIT,
        "llama_cpp_commit": LLAMA_CPP_SUBMODULE_COMMIT,
        "cuda_architecture": "89",
        "requested": list(ROOT_REQUIREMENTS),
        "files": rows,
    }
    (final / "candidate-lock.json").write_text(json.dumps(manifest))
    return final, rows


def venv(root):
    from types import SimpleNamespace
    return SimpleNamespace(
        interpreter_in_project_venv=True, project_root=str(root),
        interpreter=str(root / ".venv/bin/python"),
    )


class Pypi:
    def __init__(self, rows, *, bad=None, yanked=None):
        self.rows = {normalize_project_name(x["name"]): x for x in rows}
        self.calls = []
        self.bad = bad
        self.yanked = yanked

    def __call__(self, name, version):
        key = normalize_project_name(name)
        self.calls.append(key)
        row = self.rows[key]
        data = {
            "filename": row["filename"],
            "packagetype": "bdist_wheel",
            "digests": {"sha256": row["sha256"]},
            "size": row["size_bytes"],
            "yanked": self.yanked == key,
        }
        if self.bad == key:
            data["digests"]["sha256"] = "0" * 64
        return {"urls": [data]}


class PipDryRun:
    def __init__(self, rows, *, missing=False, fail=False):
        self.rows = rows
        self.calls = []
        self.missing = missing
        self.fail = fail

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        assert argv[:4] == [
            str(Path(kwargs["cwd"]) / ".venv/bin/python"),
            "-m", "pip", "--isolated",
        ]
        assert "--dry-run" in argv
        assert "--ignore-installed" in argv
        assert "--no-index" in argv
        assert "--no-cache-dir" in argv
        assert "download" not in argv
        assert "--no-deps" not in argv
        assert "llama-cpp-python" not in " ".join(argv)
        assert kwargs.get("timeout")
        if self.fail:
            return subprocess.CompletedProcess(argv, 1, "", "resolver failure")
        selected = self.rows[:-1] if self.missing else self.rows
        manifest = {"install": [
            {"metadata": {"name": r["name"], "version": r["version"]}}
            for r in selected
        ]}
        Path(argv[argv.index("--report") + 1]).write_text(json.dumps(manifest))
        return subprocess.CompletedProcess(argv, 0, "", "")


class CandidateAuditTests(unittest.TestCase):
    def invoke(self, root, *, source=None, pip=None, authorized=True, environment=None):
        _, rows = candidate_fixture(root)
        py = source or Pypi(rows)
        resolver = pip or PipDryRun(rows)
        report = audit_candidate(
            project_root=root, environment=environment or venv(root),
            pypi_metadata=py, run=resolver, network_authorized=authorized,
        )
        return report, py, resolver

    def test_independent_provenance_and_offline_closure_pass_but_no_install(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report, pypi, pip = self.invoke(root)
            self.assertEqual(report.status, "security_review_required", report.to_dict())
            self.assertEqual(report.files_verified, 8)
            self.assertEqual(len(pypi.calls), 8)
            self.assertEqual(len(pip.calls), 1)
            self.assertTrue(report.wheel_hashes_verified)
            self.assertTrue(report.pypi_metadata_verified)
            self.assertTrue(report.dependency_closure_verified)
            self.assertTrue(report.wheel_compatibility_verified)
            self.assertFalse(report.lock_approved)
            self.assertFalse(report.security_review_passed)
            self.assertFalse(report.installation_started)
            self.assertFalse(report.build_started)
            self.assertFalse(report.fallback_automatically_selected)

    def test_not_authorized_never_probes_pypi_or_pip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, rows = candidate_fixture(root)
            def forbidden(*a, **k):
                self.fail("No network or pip without authorization")
            report = audit_candidate(
                project_root=root, environment=venv(root),
                network_authorized=False, pypi_metadata=forbidden, run=forbidden)
            self.assertEqual(report.status, "metadata_network_not_authorized")

    def test_wheel_tampering_detected_before_network(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, rows = candidate_fixture(root)
            with (path / "wheels" / rows[0]["filename"]).open("ab") as output:
                output.write(b"altered")
            def forbidden(*a, **k):
                self.fail("No downstream probes for tampered wheel")
            report = audit_candidate(
                project_root=root, environment=venv(root),
                pypi_metadata=forbidden, run=forbidden, network_authorized=True)
            self.assertEqual(report.status, "invalid_wheels")

    def test_wrong_pypi_digest_blocks_without_pip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, rows = candidate_fixture(root)
            source = Pypi(rows, bad="numpy")
            report = audit_candidate(
                project_root=root, environment=venv(root),
                pypi_metadata=source,
                run=lambda *a, **k: self.fail("No pip after mismatch"),
                network_authorized=True,
            )
            self.assertEqual(report.status, "pypi_mismatch")
            self.assertFalse(report.dependency_closure_verified)

    def test_yanked_artifact_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, rows = candidate_fixture(root)
            report = audit_candidate(
                project_root=root, environment=venv(root),
                pypi_metadata=Pypi(rows, yanked="diskcache"),
                run=lambda *a, **k: self.fail("no pip"),
                network_authorized=True,
            )
            self.assertEqual(report.status, "pypi_mismatch")

    def test_pip_transitive_resolution_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, rows = candidate_fixture(root)
            report = audit_candidate(
                project_root=root, environment=venv(root),
                pypi_metadata=Pypi(rows), run=PipDryRun(rows, fail=True),
                network_authorized=True,
            )
            self.assertEqual(report.status, "dependency_closure_failed")
            self.assertTrue(report.pypi_metadata_verified)
            self.assertFalse(report.dependency_closure_verified)

    def test_partial_dry_run_report_does_not_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, rows = candidate_fixture(root)
            report = audit_candidate(
                project_root=root, environment=venv(root),
                pypi_metadata=Pypi(rows), run=PipDryRun(rows, missing=True),
                network_authorized=True,
            )
            self.assertEqual(report.status, "dependency_closure_failed")
            self.assertFalse(report.lock_approved)

    def test_wheel_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            final, rows = candidate_fixture(root)
            path = final / "wheels" / rows[0]["filename"]
            path.unlink()
            path.symlink_to(final / "candidate-lock.json")
            report = audit_candidate(
                project_root=root, environment=venv(root),
                network_authorized=True,
                pypi_metadata=lambda *a: self.fail("must not fetch"),
            )
            self.assertEqual(report.status, "invalid_wheels")

    def test_changed_upstream_pin_blocks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            final, rows = candidate_fixture(root)
            fp = final / "candidate-lock.json"
            payload = json.loads(fp.read_text())
            payload["source_commit"] = "0" * 40
            fp.write_text(json.dumps(payload))
            report = audit_candidate(
                project_root=root, environment=venv(root),
                network_authorized=True, pypi_metadata=lambda *a: self.fail("no fetch"),
            )
            self.assertEqual(report.status, "invalid_manifest")

    def test_bad_venv_blocks_before_pypi(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate_fixture(root)
            fake = venv(root)
            fake.interpreter = "/usr/bin/python3"
            report = audit_candidate(
                project_root=root, environment=fake,
                network_authorized=True,
                pypi_metadata=lambda *a: self.fail("no fetch"),
            )
            self.assertEqual(report.status, "blocked_venv")


if __name__ == "__main__":
    unittest.main()
