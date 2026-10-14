"""A portable profile submission never leaks machine identity or bypasses consent."""
import json
from io import StringIO
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from ruth_runtime.profile_contribution import (
    build_proposal, publish_proposal_pr, validate_submitted_proposal,
    sync_profile_to_git, sync_owner_profile_if_authenticated,
)
from ruth_runtime.model_catalog import select_model
from ruth_runtime.profile_registry import SCHEMA
from ruth_runtime.launcher import main


class ProfileContributionTests(unittest.TestCase):
    def prepare(self, root):
        path = root / ".ruth-build/profiles/models"
        path.mkdir(parents=True, exist_ok=True)
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

    def test_context_kv_and_compute_profile_survive_sanitized_git_roundtrip(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            local = self.prepare(root)
            choice = select_model("production")
            data = json.loads(local.read_text())
            data["model"].update({
                "gguf_sha256": choice.expected_sha256,
                "source_repository": choice.repo_id,
                "source_revision": choice.revision,
                "version": choice.version,
                "quantization": choice.quantization,
            })
            runtime_key = next(iter(data["runtime_measurements"]))
            data["context_profiles"] = {
                runtime_key + "/n_ctx=128/flash=unknown/k=unknown/v=unknown": {
                    "n_ctx_requested": 128, "kv_size_native": [256],
                    "kv_padding": None, "flash_attention": None,
                    "kv_type_k": None, "kv_type_v": None,
                    "gpu_kv_buffer_mib": 12.5,
                    "gpu_compute_buffer_mib": 56.0,
                    "backend": "llama.cpp",
                    "verified_native_allocations_only": True,
                    "inference_verified": False,
                },
            }
            local.write_text(json.dumps(data))
            proposal = build_proposal(local, project_root=root)
            self.assertEqual(proposal["proposal"]["context_profiles"][0]
                             ["gpu_kv_buffer_mib"], 12.5)
            self.assertIsNone(proposal["proposal"]["context_profiles"][0]
                              ["kv_padding"])
            public = root / proposal["repository_path"].replace(
                "profiles/submissions/", "profiles/measurements/", 1)
            public.parent.mkdir(parents=True)
            public.write_text(json.dumps(proposal["proposal"]))
            self.assertEqual(
                validate_submitted_proposal(public)["status"],
                "submitted_profile_schema_verified_review_required")
            tampered = json.loads(public.read_text())
            tampered["context_profiles"][0]["inference_verified"] = True
            public.write_text(json.dumps(tampered))
            with self.assertRaises(ValueError):
                validate_submitted_proposal(public)

    def test_proposal_rejects_temporary_profile_outside_project(self):
        # CLI orchestration tests mock build_proposal because their fixtures
        # live in temporary roots; real runtime must stay project-root-only.
        with TemporaryDirectory() as tmp:
            outside = self.prepare(Path(tmp))
            with self.assertRaisesRegex(ValueError, "Only a local Ruth memory"):
                build_proposal(outside)

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

    def test_ci_validates_owner_direct_measurements_with_same_pinned_schema(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            local = self.prepare(root)
            choice = select_model("production")
            data = json.loads(local.read_text())
            data["model"].update({
                "gguf_sha256": choice.expected_sha256,
                "source_repository": choice.repo_id,
                "source_revision": choice.revision,
                "version": choice.version,
                "quantization": choice.quantization,
            })
            local.write_text(json.dumps(data))
            proposal = build_proposal(local, project_root=root)
            measured = root / proposal["repository_path"].replace(
                "profiles/submissions/", "profiles/measurements/", 1)
            measured.parent.mkdir(parents=True)
            measured.write_text(json.dumps(proposal["proposal"]))
            result = validate_submitted_proposal(measured)
            self.assertEqual(result["status"],
                             "submitted_profile_schema_verified_review_required")
            self.assertFalse(result["production_qualified"])

    def test_cli_publication_only_uses_saved_profile_without_recording(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = self.prepare(root)
            expected = {
                "status": "git_cli_missing", "git_updated": False,
                "retryable": True,
            }
            matrix = {
                "status": "profile_catalog_ready", "profiles": [],
                "gpu_allocations_executed": False,
            }
            with patch("sys.stdout", new_callable=StringIO) as output, \
                 patch("ruth_runtime.profile_contribution.build_proposal",
                       return_value={"proposal": {"test": True},
                                     "repository_path": "profiles/submissions/test.json",
                                     "content_digest": "a" * 64}), \
                 patch("ruth_runtime.model_memory_profiles.build_profile_matrix",
                       return_value=matrix), \
                 patch("ruth_runtime.profile_contribution.sync_profile_to_git",
                       return_value=expected) as publish, \
                 patch("ruth_runtime.profile_registry.record_local_profile",
                       side_effect=AssertionError("Do not rerecord")):
                code = main(["models", "profiles", "--publish-pr",
                             "--consent-publication", "--profile-file", str(profile)])
            self.assertEqual(code, 2)
            result = json.loads(output.getvalue())
            self.assertEqual(result["status"], "profile_catalog_ready")
            self.assertTrue(result["local_profile_retained"])
            self.assertEqual(result["git_publication"]["status"], "git_cli_missing")
            self.assertTrue(publish.called)

    def test_cli_local_profile_survives_failed_publication(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = self.prepare(root)
            matrix = {"status": "profile_catalog_ready", "profiles": [],
                      "gpu_allocations_executed": False}
            with patch("sys.stdout", new_callable=StringIO) as output, \
                 patch("ruth_runtime.profile_contribution.build_proposal",
                       return_value={"proposal": {"test": True},
                                     "repository_path": "profiles/submissions/test.json",
                                     "content_digest": "a" * 64}), \
                 patch("ruth_runtime.model_memory_profiles.build_profile_matrix",
                       return_value=matrix), \
                 patch("ruth_runtime.profile_registry.record_local_profile",
                       return_value={"status": "local_profile_updated",
                                     "profile_file": str(profile)}) as record, \
                 patch("ruth_runtime.profile_contribution.sync_profile_to_git",
                       return_value={"status": "git_cli_missing",
                                     "git_updated": False}):
                code = main(["models", "profiles", "--host",
                             "--evidence-file", str(root / "probe.json"),
                             "--record-local", "--publish-pr",
                             "--consent-publication"])
            self.assertEqual(code, 2)
            result = json.loads(output.getvalue())
            self.assertEqual(result["status"], "profile_catalog_ready")
            self.assertEqual(result["local_profile_update"]["status"],
                             "local_profile_updated")
            self.assertEqual(result["git_publication"]["status"], "git_cli_missing")
            self.assertTrue(result["local_profile_retained"])
            self.assertEqual(record.call_count, 1)

    def test_cli_implicit_owner_sync_is_not_additional_authorization(self):
        with TemporaryDirectory() as tmp:
            profile = self.prepare(Path(tmp))
            matrix = {"status": "profile_catalog_ready", "profiles": []}
            with patch("sys.stdout", new_callable=StringIO) as output, \
                 patch("ruth_runtime.profile_contribution.build_proposal",
                       return_value={"proposal": {"test": True},
                                     "repository_path": "profiles/submissions/test.json",
                                     "content_digest": "a" * 64}), \
                 patch("ruth_runtime.model_memory_profiles.build_profile_matrix",
                       return_value=matrix), \
                 patch("ruth_runtime.profile_registry.record_local_profile",
                       return_value={"status": "local_profile_updated",
                                     "profile_file": str(profile)}), \
                 patch("ruth_runtime.profile_contribution.sync_owner_profile_if_authenticated",
                       return_value={"status": "git_profile_updated_model_profiling",
                                     "git_updated": True, "branch": "model-profiling"}) as sync:
                code = main(["models", "profiles", "--host", "--record-local",
                             "--evidence-file", "/tmp/measurement.json"])
            self.assertEqual(code, 0)
            self.assertTrue(sync.called)
            body = json.loads(output.getvalue())
            self.assertEqual(body["git_publication"]["status"],
                             "git_profile_updated_model_profiling")

    def test_cli_owner_can_explicitly_sync_without_consent_flag(self):
        with TemporaryDirectory() as tmp:
            profile = self.prepare(Path(tmp))
            matrix = {"status": "profile_catalog_ready", "profiles": []}
            with patch("sys.stdout", new_callable=StringIO) as output, \
                 patch("ruth_runtime.profile_contribution.build_proposal",
                       return_value={"proposal": {"test": True},
                                     "repository_path": "profiles/submissions/test.json",
                                     "content_digest": "a" * 64}), \
                 patch("ruth_runtime.model_memory_profiles.build_profile_matrix",
                       return_value=matrix), \
                 patch("ruth_runtime.profile_contribution.sync_profile_to_git",
                       return_value={"status": "git_profile_updated_model_profiling",
                                     "git_updated": True, "branch": "model-profiling"}) as sync:
                code = main(["models", "profiles", "--profile-file", str(profile),
                             "--sync-git"])
            self.assertEqual(code, 0)
            self.assertTrue(sync.called)
            self.assertTrue(json.loads(output.getvalue())["git_publication"]["git_updated"])

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
        self.assertIsNone(result["review_required"])

    def test_tampered_digest_blocked_before_auth(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            proposal = build_proposal(self.prepare(root), project_root=root)
        proposal["proposal"]["runtime_measurements"][0]["weights_buffer_mib_max"] = 123.0
        with self.assertRaises(ValueError):
            publish_proposal_pr(proposal, permit_external_publication=True,
                                run=lambda *args, **kwargs: self.fail("No network"))

    def test_authenticated_owner_commits_directly_on_model_profiling(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            proposal = build_proposal(self.prepare(root), project_root=root)
        calls = []
        def run(args, **kwargs):
            calls.append(args)
            line = " ".join(args)
            if "api user --jq .login" in line:
                return SimpleNamespace(returncode=0, stdout="yannericetchart-dev",
                                       stderr="")
            if "?ref=model-profiling" in line:
                return SimpleNamespace(returncode=1, stdout="",
                                       stderr="gh: Not Found (HTTP 404)")
            if ("--method PUT" in line and
                "repos/yannericetchart-dev/ruth/contents/profiles/measurements" in line):
                endpoint = next(arg for arg in args if arg.startswith(
                    "repos/yannericetchart-dev/ruth/contents/"))
                path = endpoint.split("/contents/", 1)[1]
                response = {"content": {"path": path},
                            "commit": {"sha": "f" * 40}}
                return SimpleNamespace(returncode=0, stdout=json.dumps(response),
                                       stderr="")
            self.fail("Unexpected GitHub command: " + line)
        result = sync_profile_to_git(proposal, run=run)
        self.assertEqual(result["status"], "git_profile_updated_model_profiling")
        self.assertTrue(result["git_updated"])
        self.assertEqual(result["branch"], "model-profiling")
        self.assertFalse(result["review_required"])
        self.assertTrue(result["repository_path"].startswith("profiles/measurements/"))
        self.assertFalse(any("pulls" in " ".join(x) for x in calls))
        self.assertFalse(any("fork" in " ".join(x) for x in calls))
        self.assertFalse(any("repos/contributor" in " ".join(x) for x in calls))

    def test_owner_idempotent_existing_measurement(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            proposal = build_proposal(self.prepare(root), project_root=root)
        content = json.dumps(proposal["proposal"]).encode("utf-8")
        import base64
        calls = []
        def run(args, **kwargs):
            calls.append(args)
            if args[:4] == ["gh", "api", "user", "--jq"]:
                return SimpleNamespace(returncode=0, stdout="yannericetchart-dev",
                                       stderr="")
            if "?ref=model-profiling" in args[-1]:
                return SimpleNamespace(returncode=0, stdout=json.dumps({
                    "type": "file",
                    "content": base64.b64encode(content).decode("ascii"),
                }), stderr="")
            self.fail("Owner must not overwrite unchanged measurements")
        result = sync_profile_to_git(proposal, run=run)
        self.assertEqual(result["status"], "git_profile_already_current")
        self.assertFalse(result["git_updated"])
        self.assertEqual(len(calls), 2)

    def test_owner_automatic_sync_requires_authenticated_identity(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            proposal = build_proposal(self.prepare(root), project_root=root)
        calls = []
        def outsider(args, **kwargs):
            calls.append(args)
            return SimpleNamespace(returncode=0, stdout="contributor",
                                   stderr="")
        result = sync_owner_profile_if_authenticated(proposal, run=outsider)
        self.assertEqual(result["status"], "git_owner_review_required")
        self.assertFalse(result["git_updated"])
        self.assertEqual(len(calls), 1)

    def test_owner_direct_write_does_not_hide_api_errors(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            proposal = build_proposal(self.prepare(root), project_root=root)
        def run(args, **kwargs):
            if "api user --jq .login" in " ".join(args):
                return SimpleNamespace(returncode=0, stdout="yannericetchart-dev",
                                       stderr="")
            return SimpleNamespace(returncode=1, stdout="",
                                   stderr="gh: Forbidden (HTTP 403)")
        with self.assertRaisesRegex(RuntimeError, "lookup failed"):
            sync_profile_to_git(proposal, run=run)

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
