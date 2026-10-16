"""Official Ruth entrypoint: sequential read-only checks or targeted inspections."""
from collections.abc import Sequence
from pathlib import Path
import sys

from environment import plan as plan_dependencies
from environment.diagnostics import inspect as inspect_diagnostics
from environment.runtime.contracts import DependencyPlan, PythonEnvironment
from runtime import inspect_environment

# Initial smoke check only. Ruth's full dependency set has not been approved.
DEFAULT_CHECK_PACKAGES = ("pip",)


def _check_python(root: Path) -> PythonEnvironment:
    """Run the first step and retain its report for the last diagnostic step."""
    print("Ruth: avvio controllo ambiente Python.")
    report = inspect_environment(root)
    if report.interpreter_in_project_venv:
        print("Ruth: ambiente Python conforme (virtualenv isolato).")
    else:
        print("Ruth: ambiente Python non conforme o non attivo.")
        for reason in report.reasons:
            print(f"Ruth: motivo: {reason}")
    return report


def _check_dependencies(requirements: Sequence[str], root: Path) -> tuple[int, DependencyPlan | None]:
    """Run the second step; return its report without installing packages."""
    print("Ruth: avvio controllo dipendenze Python.")
    try:
        report = plan_dependencies(requirements, project_root=root)
    except ValueError as error:
        print(f"Ruth: requisito non valido: {error}")
        return 2, None
    if not report.python.interpreter_in_project_venv:
        print("Ruth: ambiente Python non conforme; controllo pacchetti bloccato.")
        for reason in report.python.reasons:
            print(f"Ruth: motivo: {reason}")
    for package in report.packages:
        detail = f" (versione {package.installed_version})" if package.installed_version else ""
        print(f"Ruth: pacchetto {package.name}: {package.status}{detail}")
    if report.all_declared_present:
        print("Ruth: pacchetti dichiarati presenti (compatibilita' non verificata).")
        return 0, report
    print("Ruth: uno o piu' pacchetti non sono confermati.")
    return 1, report


def _diagnose(python: PythonEnvironment, dependencies: DependencyPlan | None) -> None:
    """Third step uses existing results only, even on a failed prerequisite."""
    print("Ruth: avvio diagnostica ambiente.")
    report = inspect_diagnostics(python, dependencies)
    for issue in report.issues:
        target = f" [{issue.package}]" if issue.package else ""
        print(f"Ruth: diagnostica {issue.code}{target}: {issue.message}")
        print(f"Ruth: suggerimento: {issue.suggested_action}")
    print(f"Ruth: diagnostica: {report.status} (nessuna riparazione automatica).")


def main(argv: Sequence[str] = ()) -> int:
    """Run PY -> DEP -> DIAG by default, preserving explicit DEP inspection."""
    args = list(argv)
    root = Path(__file__).resolve().parent
    if not args:
        print("Ruth: sequenza automatica dei controlli integrati.")
        python = _check_python(root)
        if not python.interpreter_in_project_venv:
            print("Ruth: controllo dipendenze saltato: ambiente Python non conforme.")
            _diagnose(python, None)
            print("Ruth: sequenza controlli terminata con problemi.")
            return 1
        print("Ruth: dipendenze minime da controllare: pip (non elenco completo di Ruth).")
        code, dep = _check_dependencies(DEFAULT_CHECK_PACKAGES, root)
        _diagnose(python, dep)
        if code == 0:
            print("Ruth: sequenza controlli completata; compatibilita' runtime non verificata.")
        else:
            print("Ruth: sequenza controlli terminata con problemi.")
        return code

    if args[0] != "dependencies" or len(args) < 2:
        print("Ruth: uso: python main.py [dependencies PACCHETTO [PACCHETTO==VERSIONE ...]]")
        return 2
    return _check_dependencies(args[1:], root)[0]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
