"""No-network tests for the secure Kaggle API token flow."""

from contextlib import redirect_stdout, redirect_stderr
import io
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from model import kaggle_auth


class KaggleAuthTests(unittest.TestCase):
    def test_validation_uses_kagglehub_and_clears_in_memory_credentials(self):
        with (
            patch("kagglehub.config.set_kaggle_api_token") as set_token,
            patch("kagglehub.config.clear_kaggle_credentials") as clear,
            patch("kagglehub.whoami",
                  return_value={"username": "ruth-tester"}) as whoami,
        ):
            self.assertTrue(kaggle_auth._validate_token("KGAT_fake"))
        set_token.assert_called_once_with("KGAT_fake")
        whoami.assert_called_once_with(verbose=False)
        clear.assert_called_once()

    def test_validation_failure_also_clears_in_memory_credentials(self):
        with (
            patch("kagglehub.config.set_kaggle_api_token"),
            patch("kagglehub.config.clear_kaggle_credentials") as clear,
            patch("kagglehub.whoami",
                  side_effect=RuntimeError("private diagnostics")),
        ):
            with self.assertRaises(RuntimeError):
                kaggle_auth._validate_token("KGAT_fake")
        clear.assert_called_once()

    def test_existing_environment_token_is_reused_without_prompt(self):
        output = io.StringIO()
        with (
            patch.dict(os.environ, {"KAGGLE_API_TOKEN": "private"}, clear=True),
            patch("model.kaggle_auth.getpass") as prompt,
            patch("model.kaggle_auth._validate_token") as verify,
            redirect_stdout(output),
        ):
            result = kaggle_auth.configure_personal_token(interactive=True)
        self.assertEqual(result, "existing")
        prompt.assert_not_called()
        verify.assert_not_called()
        self.assertNotIn("private", output.getvalue())

    def test_existing_token_file_skips_prompt(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "access_token"
            path.write_text("existing-value")
            with (
                patch.dict(os.environ, {}, clear=True),
                patch("model.kaggle_auth._credential_path", return_value=path),
                patch("model.kaggle_auth.getpass") as prompt,
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    kaggle_auth.configure_personal_token(interactive=True),
                    "existing",
                )
            prompt.assert_not_called()

    def test_noninteractive_setup_never_requests_a_token(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("model.kaggle_auth._existing_credentials", return_value=False),
            patch("model.kaggle_auth.getpass") as prompt,
            redirect_stdout(io.StringIO()),
        ):
            result = kaggle_auth.configure_personal_token(interactive=False)
        self.assertEqual(result, "noninteractive")
        prompt.assert_not_called()

    def test_hidden_input_validation_and_user_only_permissions(self):
        secret = "KGAT_local-private-value"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / ".kaggle" / "access_token"
            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                patch.dict(os.environ, {}, clear=True),
                patch("model.kaggle_auth._credential_path", return_value=path),
                patch("model.kaggle_auth.getpass", return_value=secret) as prompt,
                patch("model.kaggle_auth._validate_token", return_value=True) as verify,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                result = kaggle_auth.configure_personal_token(interactive=True)
            self.assertEqual(result, "configured")
            prompt.assert_called_once()
            verify.assert_called_once_with(secret)
            self.assertEqual(path.read_text(), secret + "\n")
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(path.parent.stat().st_mode), 0o700)
            self.assertNotIn(secret, stdout.getvalue() + stderr.getvalue())

    def test_validation_failure_preserves_previous_token(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / ".kaggle" / "access_token"
            path.parent.mkdir()
            path.write_text("old-token\n")
            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                patch.dict(os.environ, {}, clear=True),
                patch("model.kaggle_auth._credential_path", return_value=path),
                patch("model.kaggle_auth.getpass",
                      return_value="invalid-secret"),
                patch("model.kaggle_auth._validate_token", return_value=False),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                result = kaggle_auth.configure_personal_token(
                    force=True, interactive=True
                )
            self.assertEqual(result, "failed")
            self.assertEqual(path.read_text(), "old-token\n")
            self.assertNotIn("invalid-secret", stdout.getvalue() + stderr.getvalue())

    def test_verification_exception_does_not_echo_secret(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as temp:
            with (
                patch.dict(os.environ, {}, clear=True),
                patch("model.kaggle_auth._credential_path",
                      return_value=Path(temp) / "access_token"),
                patch("model.kaggle_auth.getpass", return_value="sensitive"),
                patch("model.kaggle_auth._validate_token",
                      side_effect=ValueError("sensitive invalid")),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                result = kaggle_auth.configure_personal_token(interactive=True)
        self.assertEqual(result, "failed")
        self.assertNotIn("sensitive", stdout.getvalue() + stderr.getvalue())

    def test_blank_token_and_interrupted_prompt_skip(self):
        for value in ("  ", EOFError(), KeyboardInterrupt()):
            with self.subTest(value=type(value).__name__):
                with (
                    patch.dict(os.environ, {}, clear=True),
                    patch("model.kaggle_auth._existing_credentials", return_value=False),
                    patch("model.kaggle_auth.getpass",
                          side_effect=value if isinstance(value, BaseException)
                          else None, return_value=value if isinstance(value, str)
                          else None),
                    redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(
                        kaggle_auth.configure_personal_token(interactive=True),
                        "skipped",
                    )

    def test_repository_local_cache_is_refused(self):
        root = Path(kaggle_auth.__file__).resolve().parents[1]
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("model.kaggle_auth._credential_path",
                  return_value=root / "access_token"),
            patch("model.kaggle_auth.getpass") as prompt,
            redirect_stderr(io.StringIO()),
        ):
            result = kaggle_auth.configure_personal_token(
                force=True, interactive=True
            )
        self.assertEqual(result, "unsafe_cache")
        prompt.assert_not_called()

    def test_force_can_replace_an_existing_token(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / ".kaggle" / "access_token"
            path.parent.mkdir()
            path.write_text("old\n")
            with (
                patch.dict(os.environ, {}, clear=True),
                patch("model.kaggle_auth._credential_path", return_value=path),
                patch("model.kaggle_auth.getpass", return_value="new"),
                patch("model.kaggle_auth._validate_token", return_value=True),
                redirect_stdout(io.StringIO()),
            ):
                result = kaggle_auth.configure_personal_token(
                    force=True, interactive=True
                )
            self.assertEqual(result, "configured")
            self.assertEqual(path.read_text(), "new\n")

    def test_cli_failure_exit_code(self):
        with patch("model.kaggle_auth.configure_personal_token",
                   return_value="failed") as configure:
            self.assertEqual(kaggle_auth.main(["--force"]), 1)
            configure.assert_called_once_with(force=True)
        with patch("model.kaggle_auth.configure_personal_token",
                   return_value="skipped"):
            self.assertEqual(kaggle_auth.main([]), 0)


if __name__ == "__main__":
    unittest.main()
