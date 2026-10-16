"""Official Ruth entrypoint: read-only Python environment inspection."""
from pathlib import Path

from runtime import inspect_environment


def main() -> int:
    """Report isolation of the interpreter running Ruth, without modifying it."""
    print("Ruth: avvio controllo ambiente Python.")
    report = inspect_environment(Path(__file__).resolve().parent)
    if report.interpreter_in_project_venv:
        print("Ruth: ambiente Python conforme (virtualenv isolato).")
        return 0
    print("Ruth: ambiente Python non conforme o non attivo.")
    for reason in report.reasons:
        print(f"Ruth: motivo: {reason}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
