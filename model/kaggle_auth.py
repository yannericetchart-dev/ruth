"""Optional per-user Kaggle API token setup for Ruth.

The modern Kaggle API token is stored at ~/.kaggle/access_token (or in
KAGGLE_CONFIG_DIR). Never put credentials in Ruth's checkout, Git, command-line
arguments, or terminal output. A Kaggle token NAME is not the secret value.
"""

import argparse
from getpass import getpass
import os
from pathlib import Path
import stat
import sys
import tempfile


def _credential_path() -> Path:
    folder = os.environ.get("KAGGLE_CONFIG_DIR")
    return (Path(folder).expanduser() if folder
            else Path.home() / ".kaggle") / "access_token"


def _existing_credentials() -> bool:
    # Do not read, display, or log any existing secret.
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    path = _credential_path()
    return any(p.is_file() for p in (
        path, path.with_suffix(".txt"), path.with_name("kaggle.json")
    ))


def _cache_is_safe(path: Path) -> bool:
    checkout = Path(__file__).resolve().parents[1]
    return not (
        path.parent.resolve().is_relative_to(checkout)
        or path.resolve().is_relative_to(checkout)
    )


def _validate_token(token: str) -> bool:
    """Validate the actual modern API token with Kaggle; no persistence yet."""
    import kagglehub
    from kagglehub.config import (
        clear_kaggle_credentials, set_kaggle_api_token,
    )

    try:
        set_kaggle_api_token(token)
        user = kagglehub.whoami(verbose=False)
        return isinstance(user, dict) and bool(user.get("username"))
    finally:
        # Do not leave the candidate token in kagglehub's process-global state.
        clear_kaggle_credentials()


def _save_token(token: str, path: Path) -> None:
    """Atomically save credentials with restrictive user-only permissions."""
    directory = path.parent
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    fd, temp_name = tempfile.mkstemp(prefix=".access_token-", dir=str(directory))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            os.fchmod(output.fileno(), stat.S_IRUSR | stat.S_IWUSR)
            output.write(token + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_name, path)
    finally:
        Path(temp_name).unlink(missing_ok=True)


def configure_personal_token(*, force: bool = False,
                             interactive: bool | None = None) -> str:
    """Returns existing, noninteractive, skipped, configured, failed, unsafe_cache."""
    if not force and _existing_credentials():
        print("Kaggle: credenziali personali gia' disponibili.")
        return "existing"

    if interactive is None:
        interactive = bool(sys.stdin is not None and sys.stdin.isatty())
    if not interactive:
        print(
            "Kaggle: setup non interattivo; inserimento token saltato. "
            "Usa KAGGLE_API_TOKEN in CI oppure "
            ".venv/bin/python -m model.kaggle_auth --force da terminale."
        )
        return "noninteractive"

    path = _credential_path()
    if not _cache_is_safe(path):
        print(
            "Kaggle: percorso delle credenziali dentro Ruth. "
            "Imposta KAGGLE_CONFIG_DIR fuori dal repository e riprova.",
            file=sys.stderr,
        )
        return "unsafe_cache"

    print(
        "Kaggle: incolla il VALORE segreto del token personale, non il nome "
        "(ad esempio ruth-local).\n"
        "Crea/gestisci i token su https://www.kaggle.com/settings/api\n"
        "Invio salta la configurazione. Il token non verra' salvato in Git."
    )
    try:
        token = getpass("Incolla il tuo KAGGLE_API_TOKEN (Invio per saltare): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nConfigurazione Kaggle saltata.")
        return "skipped"
    if not token:
        print("Nessun token Kaggle inserito: configurazione saltata.")
        return "skipped"

    try:
        if not _validate_token(token):
            print(
                "Kaggle: token non verificato. Controlla connessione e permessi.",
                file=sys.stderr,
            )
            return "failed"
        _save_token(token, path)
    except Exception:
        # Never emit exception content: libraries may include request credentials.
        print(
            "Kaggle: impossibile verificare o salvare il token personale. "
            "Controlla connessione, permessi e riprova.",
            file=sys.stderr,
        )
        return "failed"
    finally:
        token = None

    if os.environ.get("KAGGLE_API_TOKEN"):
        print(
            "Kaggle: attenzione, KAGGLE_API_TOKEN nell'ambiente ha "
            "precedenza sul token locale."
        )
    print("Kaggle: token verificato e salvato nella cartella personale protetta.")
    return "configured"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Configura il token personale Kaggle per Ruth."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Sostituisce il token locale (se valido), senza stampare segreti.",
    )
    args = parser.parse_args(argv)
    outcome = configure_personal_token(force=args.force)
    return 1 if outcome in ("failed", "unsafe_cache") else 0


if __name__ == "__main__":
    raise SystemExit(main())
