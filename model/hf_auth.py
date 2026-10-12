"""Optional per-user Hugging Face authentication during Ruth setup.

Token input is hidden by getpass; the Hugging Face library validates and
stores credentials in its user cache, never in the repository or Git config.
Do not accept secrets as command-line arguments or print exception bodies.
"""

import argparse
from getpass import getpass
from pathlib import Path
import sys

from huggingface_hub import get_token, login
from huggingface_hub import constants


def configure_personal_token(*, force: bool = False,
                             interactive: bool | None = None) -> str:
    """Prompt once on interactive setup; otherwise keep public downloads working.

    Returns one of existing, noninteractive, skipped, configured, failed,
    unsafe_cache. Never returns the token.
    """
    if not force and get_token():
        print("Hugging Face: credenziali personali gia' disponibili.")
        return "existing"

    if interactive is None:
        interactive = bool(sys.stdin is not None and sys.stdin.isatty())
    if not interactive:
        print(
            "Hugging Face: setup non interattivo; autenticazione saltata. "
            "Configura il tuo token con .venv/bin/python -m model.hf_auth."
        )
        return "noninteractive"

    # HF_HOME can be changed by the user. Reject any location under Ruth's
    # checkout so the credential never ends up inside the Git repository,
    # even if a future .gitignore changes.
    checkout = Path(__file__).resolve().parents[1]
    token_path = Path(constants.HF_TOKEN_PATH).expanduser().resolve()
    if token_path.is_relative_to(checkout):
        print(
            "Hugging Face: percorso delle credenziali dentro Ruth. "
            "Imposta HF_HOME al di fuori del repository e riprova.",
            file=sys.stderr,
        )
        return "unsafe_cache"

    print(
        "Hugging Face: ogni utente usa il proprio token personale (Read).\n"
        "Crea un token: https://huggingface.co/settings/tokens\n"
        "Il token e' facoltativo per i modelli pubblici e non viene salvato in Git."
    )
    try:
        token = getpass("Incolla il tuo HF_TOKEN (Invio per saltare): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nConfigurazione HF saltata.")
        return "skipped"

    if not token:
        print("Nessun token inserito: download pubblici disponibili senza autenticazione.")
        return "skipped"

    try:
        # login(token=...) validates against Hugging Face before saving;
        # explicitly disable writing secrets into Git credential helpers.
        login(token=token, add_to_git_credential=False)
    except Exception:
        # Never include the token or raw exception in terminal/log output.
        print(
            "Token Hugging Face non verificato/salvato. "
            "Controlla token, connessione e permessi e riprova.",
            file=sys.stderr,
        )
        return "failed"
    finally:
        token = None

    print("Hugging Face: autenticazione personale configurata nella cache utente.")
    return "configured"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Configura il token personale Hugging Face per Ruth."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Sostituisce l'autenticazione gia' presente su questa macchina.",
    )
    args = parser.parse_args(argv)
    result = configure_personal_token(force=args.force)
    return 1 if result in ("failed", "unsafe_cache") else 0


if __name__ == "__main__":
    raise SystemExit(main())
