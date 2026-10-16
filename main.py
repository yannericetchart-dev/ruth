"""Official Ruth entrypoint: read-only Python and dependency inspection."""
from pathlib import Path
from collections.abc import Sequence
import sys

from runtime import inspect_environment
from environment import plan as plan_dependencies


def main(argv: Sequence[str] = ()) -> int:
    """Show a clear terminal result without changing Python or installing packages."""
    args = list(argv)
    root = Path(__file__).resolve().parent
    if not args:
        print("Ruth: avvio controllo ambiente Python.")
        report = inspect_environment(root)
        if report.interpreter_in_project_venv:
            print("Ruth: ambiente Python conforme (virtualenv isolato).")
            return 0
        print("Ruth: ambiente Python non conforme o non attivo.")
        for reason in report.reasons:
            print(f"Ruth: motivo: {reason}")
        return 1

    if args[0] != "dependencies" or len(args) < 2:
        print("Ruth: uso: python main.py [dependencies PACCHETTO [PACCHETTO==VERSIONE ...]]")
        return 2
    print("Ruth: avvio controllo dipendenze Python.")
    try:
        report = plan_dependencies(args[1:], project_root=root)
    except ValueError as error:
        print(f"Ruth: requisito non valido: {error}")
        return 2
    if not report.python.interpreter_in_project_venv:
        print("Ruth: ambiente Python non conforme; controllo pacchetti bloccato.")
        for reason in report.python.reasons:
            print(f"Ruth: motivo: {reason}")
    for package in report.packages:
        detail = f" (versione {package.installed_version})" if package.installed_version else ""
        print(f"Ruth: pacchetto {package.name}: {package.status}{detail}")
    if report.all_declared_present:
        print("Ruth: pacchetti dichiarati presenti (compatibilita' non verificata).")
        return 0
    print("Ruth: uno o piu' pacchetti non sono confermati.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
