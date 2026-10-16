"""Official Ruth entrypoint: sequential read-only checks or targeted inspections."""
from collections.abc import Sequence
from pathlib import Path
import sys

from environment import plan as plan_dependencies
from runtime import inspect_environment

# Initial smoke check only. Ruth's full dependency set has not been approved.
DEFAULT_CHECK_PACKAGES = ("pip",)


def _check_python(root: Path) -> bool:
    """Run the first integrated step and report its outcome."""
    print("Ruth: avvio controllo ambiente Python.")
    report = inspect_environment(root)
    if report.interpreter_in_project_venv:
        print("Ruth: ambiente Python conforme (virtualenv isolato).")
        return True
    print("Ruth: ambiente Python non conforme o non attivo.")
    for reason in report.reasons:
        print(f"Ruth: motivo: {reason}")
    return False


def _check_dependencies(requirements: Sequence[str], root: Path) -> int:
    """Run the second integrated step, never installing or importing packages."""
    print("Ruth: avvio controllo dipendenze Python.")
    try:
        report = plan_dependencies(requirements, project_root=root)
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


def main(argv: Sequence[str] = ()) -> int:
    """Run all integrated checks in order; explicit subcommands stay targeted."""
    args = list(argv)
    root = Path(__file__).resolve().parent
    if not args:
        print("Ruth: sequenza automatica dei controlli integrati.")
        if not _check_python(root):
            print("Ruth: controllo dipendenze saltato: ambiente Python non conforme.")
            return 1
        print("Ruth: dipendenze minime da controllare: pip (non elenco completo di Ruth).")
        result = _check_dependencies(DEFAULT_CHECK_PACKAGES, root)
        if result == 0:
            print("Ruth: sequenza controlli completata; compatibilita' runtime non verificata.")
        else:
            print("Ruth: sequenza controlli terminata con problemi.")
        return result

    if args[0] != "dependencies" or len(args) < 2:
        print("Ruth: uso: python main.py [dependencies PACCHETTO [PACCHETTO==VERSIONE ...]]")
        return 2
    return _check_dependencies(args[1:], root)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
