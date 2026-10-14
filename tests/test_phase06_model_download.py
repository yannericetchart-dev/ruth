"""Offline regression suite for explicit Hugging Face GGUF downloads."""
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ruth_runtime.launcher import main
from ruth_runtime.model_catalog import select_model
from ruth_runtime.model_download import (
    ALLOWED_DOWNLOAD_HOSTS, _SafeHfRedirect, _validate_hf_download_url,
    _manifest, _local_path, download_pinned_model,
)
from urllib.request import Request

PAYLOAD = b"GGUF" + b"R" * (1024 * 1024 + 23)


def choice():
    return replace(
        select_model("test"),
        expected_sha256=hashlib.sha256(PAYLOAD).hexdigest(),
        expected_size_bytes=len(PAYLOAD),
    )


def metadata(model, *, wrong=False, gated=False):
    return {
        "sha": model.revision,
        "cardData": {"license": model.license},
        "private": False, "gated": gated,
        "siblings": [{
            "rfilename": model.filename, "size": model.expected_size_bytes,
            "lfs": {"sha256": "0" * 64 if wrong else model.expected_sha256,
                    "size": model.expected_size_bytes},
        }],
    }


class FakeResponse(io.BytesIO):
    def __init__(self, data, *, status=200, headers=None,
                 url="https://cas-bridge.xethub.hf.co/asset"):
        super().__init__(data)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(data))}
        self.url = url

    def geturl(self):
        return self.url


class FakeDownload:
    def __init__(self, data=PAYLOAD, *, override_status=None,
                 override_url=None, override_range=None,
                 reported_length=None):
        self.data = data
        self.calls = []
        self.override_status = override_status
        self.override_url = override_url
        self.override_range = override_range
        self.reported_length = reported_length

    def __call__(self, req):
        self.calls.append(req)
        actual = req.headers.get("Range", req.headers.get("range"))
        if actual:
            start = int(actual.removeprefix("bytes=").removesuffix("-"))
            payload = self.data[start:]
            status = 206
            headers = {
                "Content-Length": str(len(payload)),
                "Content-Range": f"bytes {start}-{len(self.data)-1}/{len(self.data)}",
            }
            if self.override_range is not None:
                headers["Content-Range"] = self.override_range
        else:
            payload = self.data
            status = 200
            headers = {"Content-Length": str(len(payload))}
        if self.reported_length is not None:
            headers["Content-Length"] = str(self.reported_length)
        return FakeResponse(
            payload, status=self.override_status or status, headers=headers,
            url=self.override_url or "https://cas-bridge.xethub.hf.co/asset",
        )


def usage(*_):
    return SimpleNamespace(free=16 * 1024**3)


class ResumableModelDownloadTests(unittest.TestCase):
    def call(self, root, model=None, *, download=None, license=True,
             allow=True, fetch=None, disk=usage):
        selected = model or choice()
        return download_pinned_model(
            selected, project_root=root, allow_download=allow,
            acknowledge_model_license=license,
            fetch_metadata=fetch or (lambda _: metadata(selected)),
            open_request=download or FakeDownload(), disk_usage=disk,
        )

    def test_no_download_permission_never_calls_network_or_creates_dir(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            blocked = self.call(
                root, allow=False,
                fetch=lambda _: self.fail("No metadata"),
                download=lambda _: self.fail("No download"))
            self.assertEqual(blocked.status, "model_download_not_authorized")
            self.assertFalse((root / "models").exists())

    def test_license_not_acknowledged_blocks_before_network(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            blocked = self.call(
                root, license=False,
                fetch=lambda _: self.fail("No metadata"),
                download=lambda _: self.fail("No download"))
            self.assertEqual(blocked.status, "model_license_acknowledgement_required")
            self.assertFalse(blocked.download_started)

    def test_unpinned_production_variant_cannot_download(self):
        # Gemma Q4_K_L is now fully pinned in the catalog. Deliberately
        # remove the immutable revision in this negative test so the
        # downloader must refuse BEFORE metadata network or transfer.
        with TemporaryDirectory() as temp:
            root = Path(temp)
            incomplete = replace(select_model("production"), revision=None)
            blocked = self.call(
                root, model=incomplete,
                fetch=lambda _: self.fail("No metadata"),
                download=lambda _: self.fail("No download"))
            self.assertEqual(blocked.status, "model_metadata_not_pinned")
            self.assertFalse(blocked.download_started)
            self.assertFalse(blocked.model_verified)
            self.assertFalse((root / "models").exists())

    def test_full_download_verifies_hash_and_promotes_exact_file(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            model = choice()
            remote = FakeDownload()
            result = self.call(root, model, download=remote)
            self.assertEqual(result.status, "model_download_verified", result.to_dict())
            self.assertTrue(result.model_verified)
            self.assertTrue(result.download_started)
            self.assertTrue(result.license_acknowledged_for_this_download)
            self.assertEqual(result.bytes_present, len(PAYLOAD))
            self.assertFalse(result.model_loaded)
            self.assertFalse(result.gpu_offload_executed)
            self.assertFalse(result.inference_executed)
            final = _local_path(root, model)
            self.assertEqual(final.read_bytes(), PAYLOAD)
            manifest = json.loads((final.parent / (final.name + ".manifest.json")).read_text())
            self.assertEqual(manifest["sha256"], model.expected_sha256)
            self.assertTrue(manifest["verified"])
            self.assertEqual(len(remote.calls), 1)
            self.assertNotIn("range", remote.calls[0].headers)
            self.assertFalse(Path(str(final) + ".part").exists())

    def test_valid_complete_model_is_reused_without_network(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            model = choice()
            self.call(root, model)
            report = self.call(root, model, fetch=lambda _: self.fail("No metadata"),
                               download=lambda _: self.fail("No network"))
            self.assertEqual(report.status, "model_already_verified")
            self.assertTrue(report.model_verified)
            self.assertFalse(report.download_started)

    def test_resumes_partial_with_exact_http_206_range(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            model = choice()
            final = _local_path(root, model)
            final.parent.mkdir(parents=True)
            part = final.with_name(final.name + ".part")
            part.write_bytes(PAYLOAD[:41003])
            (final.parent / (final.name + ".part.json")).write_text(
                json.dumps(_manifest(model)))
            remote = FakeDownload()
            report = self.call(root, model, download=remote)
            self.assertEqual(report.status, "model_download_verified", report.to_dict())
            self.assertEqual(final.read_bytes(), PAYLOAD)
            self.assertEqual(remote.calls[0].get_header("Range"), "bytes=41003-")

    def test_server_ignoring_range_does_not_append_bad_data(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            model = choice()
            final = _local_path(root, model)
            final.parent.mkdir(parents=True)
            part = final.with_name(final.name + ".part")
            part.write_bytes(PAYLOAD[:83])
            (final.parent / (final.name + ".part.json")).write_text(
                json.dumps(_manifest(model)))
            failed = self.call(root, model, download=FakeDownload(override_status=200))
            self.assertEqual(failed.status, "model_download_interrupted_or_blocked")
            self.assertEqual(part.read_bytes(), PAYLOAD[:83])
            self.assertFalse(final.exists())

    def test_bad_content_range_is_refused(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            model = choice()
            final = _local_path(root, model)
            final.parent.mkdir(parents=True)
            final.with_name(final.name + ".part").write_bytes(PAYLOAD[:333])
            final.with_name(final.name + ".part.json").write_text(
                json.dumps(_manifest(model)))
            report = self.call(root, model, download=FakeDownload(
                override_range=f"bytes 1-{len(PAYLOAD)-1}/{len(PAYLOAD)}"))
            self.assertEqual(report.status, "model_download_interrupted_or_blocked")
            self.assertFalse(final.exists())

    def test_hash_mismatch_does_not_promote_download(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            model = choice()
            corrupt = PAYLOAD[:3] + b"X" + PAYLOAD[4:]
            report = self.call(root, model, download=FakeDownload(data=corrupt))
            self.assertEqual(report.status, "model_download_interrupted_or_blocked")
            self.assertFalse(_local_path(root, model).exists())
            self.assertTrue(_local_path(root, model).with_name(model.filename + ".part").exists())

    def test_same_length_but_wrong_checksum_never_promoted(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            model = choice()
            other = PAYLOAD[:-1] + b"Z"
            report = self.call(root, model, download=FakeDownload(data=other))
            self.assertEqual(report.status, "model_checksum_mismatch")
            self.assertFalse(_local_path(root, model).exists())

    def test_upstream_lfs_digest_mismatch_is_blocked_before_transfer(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            model = choice()
            report = self.call(root, model,
                fetch=lambda _: metadata(model, wrong=True),
                download=lambda _: self.fail("Must not transfer"))
            self.assertEqual(report.status, "model_source_not_verified")
            self.assertFalse((root / "models").exists())

    def test_gated_repo_refused_without_auth_workflow(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            model = choice()
            report = self.call(root, model, fetch=lambda _: metadata(model, gated="auto"),
                               download=lambda _: self.fail("No gated transfers"))
            self.assertEqual(report.status, "gated_model_not_supported")

    def test_official_hf_cdn_hostnames_are_explicitly_allowed(self):
        official = {
            "huggingface.co",
            "cas-server.xethub.hf.co",
            "cas-server.xethub-eu.hf.co",
            "transfer.xethub.hf.co",
            "transfer.xethub-eu.hf.co",
            "us.aws.cdn.hf.co",
            "us.gcp.cdn.hf.co",
            "cdn-lfs-us-1.hf.co",
            "cdn-lfs-eu-1.hf.co",
        }
        self.assertTrue(official.issubset(ALLOWED_DOWNLOAD_HOSTS))
        for host in official:
            with self.subTest(host=host):
                self.assertEqual(
                    _validate_hf_download_url("https://" + host + "/asset"),
                    host,
                )

    def test_documented_hf_cdn_response_can_download_and_pass_sha(self):
        for host in ("us.aws.cdn.hf.co", "us.gcp.cdn.hf.co",
                     "cdn-lfs-eu-1.hf.co", "cas-server.xethub-eu.hf.co"):
            with self.subTest(host=host), TemporaryDirectory() as temp:
                model = choice()
                result = self.call(
                    Path(temp), model,
                    download=FakeDownload(
                        override_url="https://" + host + "/test-signed-file"
                    ),
                )
                self.assertEqual(result.status, "model_download_verified")
                self.assertTrue(result.model_verified)

    def test_redirect_handler_enforces_exact_https_fqdn(self):
        original = Request("https://huggingface.co/ns/repo/resolve/file.gguf")
        handler = _SafeHfRedirect()
        for host in ("us.aws.cdn.hf.co", "cdn-lfs-us-1.hf.co",
                     "cas-server.xethub-eu.hf.co"):
            with self.subTest(host=host):
                redirected = handler.redirect_request(
                    original, None, 302, "Found", {},
                    "https://" + host + "/test-file?signature=private"
                )
                self.assertEqual(redirected.full_url.split("/")[2], host)

    def test_external_redirects_and_downgrade_rejected_without_query_leak(self):
        bad = (
            "https://attacker.example/asset?token=SECRET",
            "https://hf.co.attacker.example/asset?token=SECRET",
            "https://us.aws.cdn.hf.co.attacker.example/asset?token=SECRET",
            "http://us.aws.cdn.hf.co/asset?token=SECRET",
            "https://someone@us.aws.cdn.hf.co/asset?token=SECRET",
            "https://us.aws.cdn.hf.co:8443/asset?token=SECRET",
            "file:///etc/passwd",
        )
        original = Request("https://huggingface.co/ns/repo/resolve/file.gguf")
        handler = _SafeHfRedirect()
        for url in bad:
            with self.subTest(url=url):
                with self.assertRaises(ValueError) as err:
                    handler.redirect_request(original, None, 302, "Found", {}, url)
                self.assertNotIn("SECRET", str(err.exception))

    def test_starting_endpoint_must_be_huggingface_co(self):
        with self.assertRaises(ValueError):
            _validate_hf_download_url(
                "https://us.aws.cdn.hf.co/asset", origin=True)
        self.assertEqual(
            _validate_hf_download_url(
                "https://huggingface.co/ns/repo/resolve/commit/file.gguf",
                origin=True),
            "huggingface.co",
        )

    def test_untrusted_redirect_url_cannot_write_model(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            model = choice()
            remote = FakeDownload(override_url="https://untrusted.example.org/gguf")
            report = self.call(root, model, download=remote)
            self.assertEqual(report.status, "model_download_interrupted_or_blocked")
            self.assertFalse(_local_path(root, model).exists())

    def test_unexpected_length_is_blocked(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            model = choice()
            report = self.call(root, model,
                               download=FakeDownload(reported_length=33))
            self.assertEqual(report.status, "model_download_interrupted_or_blocked")
            self.assertFalse(_local_path(root, model).exists())

    def test_storage_symlink_is_rejected(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            outside = root / "outside"
            outside.mkdir()
            (root / "models").symlink_to(outside, target_is_directory=True)
            report = self.call(root, fetch=lambda _: self.fail("No metadata"))
            self.assertEqual(report.status, "unsafe_or_existing_model")
            self.assertEqual(list(outside.iterdir()), [])

    def test_incomplete_local_partial_and_missing_resume_sidecar_rejected(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            model = choice()
            final = _local_path(root, model)
            final.parent.mkdir(parents=True)
            final.with_name(final.name + ".part").write_bytes(PAYLOAD[:42])
            report = self.call(root, model, download=lambda _: self.fail("No network"))
            self.assertEqual(report.status, "model_download_interrupted_or_blocked")
            self.assertFalse(final.exists())

    def test_insufficient_disk_aborts_before_network_and_preserves_resume(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            report = self.call(root, download=lambda _: self.fail("No transfer"),
                               disk=lambda _: SimpleNamespace(free=1024))
            self.assertEqual(report.status, "model_insufficient_disk")
            self.assertFalse(report.download_started)

    def test_only_explicit_cli_model_download_activates_transfer(self):
        with patch("ruth_runtime.model_download.download_pinned_model",
                   return_value=SimpleNamespace(
                       model_verified=True,
                       to_dict=lambda: {"status": "model_download_verified"})) as download:
            from io import StringIO
            with patch("sys.stdout", new_callable=StringIO) as output:
                self.assertEqual(main([
                    "models", "download", "--purpose", "test",
                    "--allow-model-download", "--accept-model-license"]), 0)
            self.assertTrue(download.call_args.kwargs["allow_download"])
            self.assertTrue(download.call_args.kwargs["acknowledge_model_license"])
            self.assertIn("model_download_verified", output.getvalue())

    def test_models_download_without_consent_is_blocked_offline(self):
        from io import StringIO
        with patch("ruth_runtime.model_catalog.fetch_huggingface_metadata",
                   side_effect=AssertionError("No HF network")), \
             patch("sys.stdout", new_callable=StringIO) as output:
            self.assertEqual(main(["models", "download", "--purpose", "test"]), 2)
            self.assertIn("model_download_not_authorized", output.getvalue())

    def test_models_download_requires_license_ack(self):
        from io import StringIO
        with patch("ruth_runtime.model_catalog.fetch_huggingface_metadata",
                   side_effect=AssertionError("No HF network")), \
             patch("sys.stdout", new_callable=StringIO) as output:
            self.assertEqual(main([
                "models", "download", "--purpose", "test", "--allow-model-download"
            ]), 2)
            self.assertIn("model_license_acknowledgement_required", output.getvalue())


if __name__ == "__main__":
    unittest.main()
