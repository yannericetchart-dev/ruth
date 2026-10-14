"""A portable profile submission never leaks machine identity or bypasses consent."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from ruth_runtime.profile_contribution import (
    build_proposal, publish_proposal_pr, validate_submitted_proposal,
)
from ruth_runtime.model_catalog import select_model
from ruth_runtime.profile_registry import SCHEMA


class ProfileContributionTests(unittest.TestCase):
    def prepare(self, root):
        path = root / ".ruth-build/profiles/models"
        path.mkdir(parents=True)
        sha = "a" * 64
        file = path / (sha + ".json")
        file.write_text(json.dumps({
            "schema": SCHEMA, "model": {
                "family": "gemma", "model": "gemma-2",
                "version": "gemma-2-9B-it", "format": "gguf",
                "quantization": "Q4_K_L", "gguf_sha256": sha,
                "source_repository": "bartowski/gemma-2-9b-it-GGUF",
                "source_revision": "b"*40,
                "gguf_serialized_bytes": 5983265728, "license": "gemma",
            },
            "runtime_measurements": {
                "ada/13.2/13.2/595.99.02/0.3.36": {
                    "architecture": "ada", "compute_capability": "8.9",
                    "tensor_core_generation": 4,
                    "cuda_backend": "13.2", "cuda_driver_max": "13.2",
                    "cuda_toolkit": "13.2", "driver_version": "595.99.02",
                    "llama_cpp_binding": "0.3.36",
                    "cuda_init_observed_nvml_delta_mib_max": 90.19,
                    "driver_reserved_mib_max": 369,
                    "weights_buffer_mib_max": 5700.31,
                    "secret_local_gpu_uuid": "GPU-do-not-publish",
                    "secret_local_user_path": "/home/secret",
                },
            },
        }))
        return file

    def test_proposal_is_canonical_and_privacy_minimized(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = build_proposal(self.prepare(root), project_root=root)
            b = build_proposal(self.prepare(root), project_root=root)
        self.assertEqual(a["content_digest"], b["content_digest"])
        self.assertIn("profiles/submissions/", a["repository_path"])
        raw = json.dumps(a["proposal"])
        self.assertNotIn("/home/", raw)
        self.assertNotIn("GPU-do-not-publish", raw)
        self.assertIn("5700.31", raw)
        self.assertFalse(a["proposal"]["production_qualified"])
        self.assertTrue(a["proposal"]["review_required"])

    def test_ci_rejects_tampered_submission_and_accepts_reviewable_schema(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            local = self.prepare(root)
            choice = select_model("production")
            data = json.loads(local.read_text())
            data["model"]["gguf_sha256"] = choice.expected_sha256
            data["model"]["source_repository"] = choice.repo_id
            data["model"]["source_revision"] = choice.revision
            data["model"]["version"] = choice.version
            data["model"]["quantization"] = choice.quantization
            local.write_text(json.dumps(data))
            proposal = build_proposal(local, project_root=root)
            public = root / proposal["repository_path"]
            public.parent.mkdir(parents=True)
            public.write_text(json.dumps(proposal["proposal"]))
            good = validate_submitted_proposal(public)
            self.assertEqual(good["status"],
                             "submitted_profile_schema_verified_review_required")
            tampered = json.loads(public.read_text())
            tampered["production_qualified"] = True
            public.write_text(json.dumps(tampered))
            with self.assertRaises(ValueError):
                validate_submitted_proposal(public)

    def test_without_consent_never_calls_external_process(self):
        result = publish_proposal_pr({}, permit_external_publication=False,
                                     run=lambda *args, **kwargs: self.fail("Network call"))
        self.assertEqual(result["status"], "git_publication_not_authorized")

    def test_missing_gh_reports_retryable_publication_without_exception(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            proposal = build_proposal(self.prepare(root), project_root=root)
        def missing(*args, **kwargs):
            raise FileNotFoundError(2, "No such file or directory", "gh")
        result = publish_proposal_pr(
            proposal, permit_external_publication=True, run=missing)
        self.assertEqual(result["status"], "git_cli_missing")
        self.assertFalse(result["git_updated"])
        self.assertTrue(result["retryable"])
        self.assertTrue(result["review_required"])

    def test_tampered_digest_blocked_before_auth(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            proposal = build_proposal(self.prepare(root), project_root=root)
        proposal["proposal"]["runtime_measurements"][0]["weights_buffer_mib_max"] = 123.0
        with self.assertRaises(ValueError):
            publish_proposal_pr(proposal, permit_external_publication=True,
                                run=lambda *args, **kwargs: self.fail("No network"))

    def test_fork_pr_flow_is_user_authenticated_not_upstream_push(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            proposal = build_proposal(self.prepare(root), project_root=root)
        calls = []
        def run(args, **kwargs):
            calls.append(args)
            line = " ".join(args)
            if "api user --jq .login" in line:
                output = "contributor"
            elif "git/ref/heads/model-profiling" in line:
                output = json.dumps({"object": {"sha": "f"*40}})
            elif "repos/yannericetchart-dev/ruth/pulls" in line:
                output = json.dumps({"html_url":
                    "https://github.com/yannericetchart-dev/ruth/pull/123"})
            else:
                output = "{}"
            if args[:3] == ["gh", "api", "repos/contributor/ruth"]:
                return SimpleNamespace(returncode=1, stdout="", stderr="Not Found")
            return SimpleNamespace(returncode=0, stdout=output, stderr="")
        result = publish_proposal_pr(proposal, permit_external_publication=True,
                                     run=run)
        self.assertEqual(result["status"], "git_profile_pull_request_opened")
        self.assertTrue(result["review_required"])
        self.assertFalse(result["git_updated"])
        self.assertTrue(all(x[0] == "gh" for x in calls))
        self.assertTrue(any("repos/contributor/ruth/contents/" in " ".join(x)
                            for x in calls))
        self.assertFalse(any("repos/yannericetchart-dev/ruth/contents/"
                             in " ".join(x) for x in calls))


if __name__ == "__main__":
    unittest.main()
