"""Pure, portable explanations of PY-001 and DEP-001 reports.

No host probing, package lookup, I/O, installation, native log processing,
or publishing machine-specific observations.
"""
from __future__ import annotations

from environment.runtime.contracts import (
    DependencyPlan, DiagnosticIssue, DiagnosticReport, PythonEnvironment,
)

# Explicit whitelist: never include raw observations in published explanations.
_PYTHON_REASONS = {
    "venv_symlink": ("virtualenv_symlink", "La .venv non deve essere un collegamento simbolico.", "Usare una .venv locale al progetto."),
    "configuration_missing": ("venv_configuration_missing", "Configurazione della .venv mancante.", "Verificare la presenza di pyvenv.cfg nella .venv."),
    "configuration_invalid": ("venv_configuration_invalid", "Configurazione di isolamento della .venv non valida.", "Controllare include-system-site-packages = false, dichiarato una sola volta."),
    "shared_site_packages": ("venv_shared_site_packages", "La .venv condivide i pacchetti di sistema.", "Configurare una .venv con site-packages di sistema disabilitati."),
    "prefix_mismatch": ("venv_prefix_mismatch", "L'interprete non risulta nella .venv del progetto.", "Avviare Ruth con l'interprete della .venv del progetto."),
    "not_virtualenv": ("python_not_virtualenv", "Python non risulta in un ambiente virtuale.", "Avviare Ruth dall'ambiente virtuale isolato del progetto."),
    "inspection_failed": ("python_inspection_failed", "Non e' stato possibile completare l'ispezione Python.", "Controllare localmente configurazione e permessi."),
    "python_too_old": ("python_version_unsupported", "Versione Python non supportata.", "Usare Python 3.10 o successivo."),
}
_PACKAGES = {
    "missing": ("package_missing", "Pacchetto dichiarato non trovato.", "Verificare localmente che la distribuzione richiesta sia disponibile."),
    "incompatible": ("package_version_mismatch", "Versione installata diversa da quella esatta richiesta.", "Confrontare la versione richiesta e quella installata nell'ambiente locale."),
    "inspection_failed": ("package_inspection_failed", "Impossibile leggere i metadati del pacchetto.", "Controllare localmente i metadati della distribuzione."),
    "blocked": ("package_check_blocked", "Ispezione del pacchetto bloccata dall'ambiente Python.", "Risolvere prima i problemi della .venv."),
}


def inspect(
    python_report: PythonEnvironment,
    dependencies: DependencyPlan | None = None,
) -> DiagnosticReport:
    """Translate known report states into actionable but non-invasive findings.

    The *existing* reports are the only inputs: never read the host.
    Ignore package versions and paths and do not guess at native failures.
    """
    issues: list[DiagnosticIssue] = []
    if not python_report.interpreter_in_project_venv:
        for reason in dict.fromkeys(python_report.reasons or ("unknown",)):
            code, message, action = _PYTHON_REASONS.get(
                reason, ("python_problem_unclassified", "Ambiente Python non conforme per una ragione non classificata.", "Esaminare localmente il report Python."),
            )
            issues.append(DiagnosticIssue(code, "error", message, action))
    if dependencies is not None:
        # The dependency report must describe the *same* observation.
        if dependencies.python != python_report:
            raise ValueError("Report Python e dipendenze non coerenti.")
        for package in dependencies.packages:
            if package.status == "present_unverified":
                continue
            code, message, action = _PACKAGES.get(
                package.status, ("package_status_unknown", "Stato del pacchetto non riconosciuto.", "Verificare il report locale."),
            )
            # Validated distribution names in DEP-001, no paths or versions.
            issues.append(DiagnosticIssue(code, "error", message, action, package.name))
    if any(item.code in ("python_inspection_failed", "package_inspection_failed") for item in issues):
        status = "inspection_incomplete"
    elif issues:
        status = "attention_required"
    else:
        status = "ok_unverified"
    return DiagnosticReport(status, tuple(issues))
