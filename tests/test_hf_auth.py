"""No-network regression tests for Ruth's per-user HF token setup."""

import io
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from model import hf_auth


class HuggingFaceAuthSetupTests(unittest.TestCase):
    def test_existing_auth_is_reused_without_prompt(self):
        stdout = io.StringIO()
        with (
            patch("model.hf_auth.get_token", return_value="hf-existing"),
            patch("model.hf_auth.getpass") as prompt,
            patch("model.hf_auth.login") as login,
            redirect_stdout(stdout),
        ):
            result = hf_auth.configure_personal_token(interactive=True)
        self.assertEqual(result, "existing")
        prompt.assert_not_called()
        login.assert_not_called()
        self.assertNotIn("hf-existing", stdout.getvalue())

    def test_headless_setup_does_not_read_stdin_or_save_token(self):
        stdout = io.StringIO()
        with (
            patch("model.hf_auth.get_token", return_value=None),
            patch("model.hf_auth.getpass") as prompt,
            patch("model.hf_auth.login") as login,
            redirect_stdout(stdout),
        ):
            result = hf_auth.configure_personal_token(interactive=False)
        self.assertEqual(result, "noninteractive")
        prompt.assert_not_called()
        login.assert_not_called()

    def test_prompt_hides_secret_and_does_not_use_git_credentials(self):
        secret = "hf-personal-private-value"
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch("model.hf_auth.get_token", return_value=None),
            patch("model.hf_auth.getpass", return_value=secret) as prompt,
            patch("model.hf_auth.login") as login,
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            result = hf_auth.configure_personal_token(interactive=True)
        self.assertEqual(result, "configured")
        prompt.assert_called_once()
        login.assert_called_once_with(
            token=secret, add_to_git_credential=False
        )
        self.assertNotIn(secret, stdout.getvalue() + stderr.getvalue())

    def test_blank_token_is_optional_for_public_models(self):
        with (
            patch("model.hf_auth.get_token", return_value=None),
            patch("model.hf_auth.getpass", return_value="   "),
            patch("model.hf_auth.login") as login,
            redirect_stdout(io.StringIO()),
        ):
            result = hf_auth.configure_personal_token(interactive=True)
        self.assertEqual(result, "skipped")
        login.assert_not_called()

    def test_invalid_token_never_echoes_exception_or_secret(self):
        secret = "hf-private-test"
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch("model.hf_auth.get_token", return_value=None),
            patch("model.hf_auth.getpass", return_value=secret),
            patch("model.hf_auth.login",
                  side_effect=ValueError("bad token: " + secret)),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            result = hf_auth.configure_personal_token(interactive=True)
        self.assertEqual(result, "failed")
        self.assertNotIn(secret, stdout.getvalue() + stderr.getvalue())
        self.assertIn("non verificato", stderr.getvalue())

    def test_ctrl_c_or_eof_skips_configuration(self):
        for error in (EOFError, KeyboardInterrupt):
            with self.subTest(error=error):
                with (
                    patch("model.hf_auth.get_token", return_value=None),
                    patch("model.hf_auth.getpass", side_effect=error),
                    redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(
                        hf_auth.configure_personal_token(interactive=True),
                        "skipped",
                    )

    def test_refuses_to_save_token_inside_repo_even_if_user_sets_hf_home(self):
        root = Path(hf_auth.__file__).resolve().parents[1]
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch("model.hf_auth.get_token", return_value=None),
            patch("model.hf_auth.constants.HF_TOKEN_PATH",
                  root / "untracked-credentials" / "token"),
            patch("model.hf_auth.getpass") as prompt,
            patch("model.hf_auth.login") as login,
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            result = hf_auth.configure_personal_token(interactive=True)
        self.assertEqual(result, "unsafe_cache")
        prompt.assert_not_called()
        login.assert_not_called()
        self.assertIn("dentro Ruth", stderr.getvalue())

    def test_force_prompts_even_if_previous_token_is_available(self):
        with (
            patch("model.hf_auth.get_token", return_value="old-token"),
            patch("model.hf_auth.getpass", return_value="new-token"),
            patch("model.hf_auth.login") as login,
            redirect_stdout(io.StringIO()),
        ):
            result = hf_auth.configure_personal_token(
                force=True, interactive=True
            )
        self.assertEqual(result, "configured")
        login.assert_called_once_with(
            token="new-token", add_to_git_credential=False
        )

    def test_cli_returns_error_only_for_auth_failure(self):
        with patch("model.hf_auth.configure_personal_token",
                   return_value="failed") as configure:
            self.assertEqual(hf_auth.main(["--force"]), 1)
            configure.assert_called_once_with(force=True)
        with patch("model.hf_auth.configure_personal_token",
                   return_value="skipped"):
            self.assertEqual(hf_auth.main([]), 0)


if __name__ == "__main__":
    unittest.main()
