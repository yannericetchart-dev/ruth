"""Official Ruth entrypoint: sequential read-only checks or targeted inspections."""
from collections.abc import Sequence
from pathlib import Path
import sys

from environment import plan as plan_dependencies
from environment.diagnostics import inspect as inspect_diagnostics
from environment.runtime.contracts import DependencyPlan, PythonEnvironment
from hardware import detect as detect_hardware
from hardware.gpu import detect as detect_gpu
from models.catalog import catalog as load_model_catalog
from models.formats.gguf import inspect_headers
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



def _check_hardware() -> None:
    """Fourth, independent step: CPU and total RAM only, no GPU probing."""
    print("Ruth: avvio inventario hardware (CPU e RAM).")
    report = detect_hardware()
    for cpu in report.devices:
        architecture = cpu.architecture or "unknown"
        logical = str(cpu.logical_processors) if cpu.logical_processors is not None else "unknown"
        print(f"Ruth: CPU: architettura {architecture}; processori logici {logical}; stato {cpu.status}.")
    memory = (str(report.memory.total_memory_bytes)
              if report.memory.total_memory_bytes is not None else "unknown")
    print(f"Ruth: RAM fisica totale (byte): {memory}; stato {report.memory.status}.")
    print(f"Ruth: inventario hardware: {report.status} (GPU e compatibilita' inferenza non verificate).")



def _check_gpu() -> None:
    """Fifth independent step: installed NVIDIA inventory, no CUDA testing."""
    print("Ruth: avvio inventario GPU NVIDIA.")
    report = detect_gpu()
    for index, gpu in enumerate(report.devices, 1):
        model = gpu.model or "unknown"
        vram = str(gpu.total_vram_bytes) if gpu.total_vram_bytes is not None else "unknown"
        print(f"Ruth: GPU NVIDIA {index}: modello {model}; VRAM totale (byte) {vram}; stato {gpu.status}.")
    print(f"Ruth: GPU NVIDIA: {report.count} rilevate; stato {report.status}.")
    print("Ruth: disponibilita' CUDA e compatibilita' inferenza non verificate.")


def _check_catalog(root: Path) -> int:
    """Sixth step: inspect only a declared local manifest, without scanning."""
    print("Ruth: avvio controllo catalogo modelli locale.")
    report = load_model_catalog(project_root=root)
    for model in report.models:
        print(f"Ruth: modello {model.id}: formato dichiarato {model.format}; stato {model.status}.")
    for reason in report.errors:
        print(f"Ruth: catalogo: problema {reason}.")
    print(f"Ruth: catalogo modelli: {report.status}; {report.count} modelli dichiarati.")
    if report.status == "not_configured":
        print("Ruth: catalogo locale non configurato (nessun download).")
    elif report.status == "empty":
        print("Ruth: nessun modello dichiarato (nessun download).")
    elif report.status == "ready":
        print("Ruth: file dichiarati presenti (formato e compatibilita' non verificati).")
    return 1 if report.status in ("partial", "invalid") else 0


def _check_gguf(root: Path) -> int:
    """Seventh step: header-only inspection of GGUF files declared in CAT."""
    print("Ruth: avvio controllo intestazioni GGUF dichiarate.")
    report = inspect_headers(project_root=root)
    for row in report.headers:
        if row.status == "header_read":
            print(f"Ruth: GGUF {row.model_id}: versione {row.version}; tensor dichiarati {row.tensor_count}; "
                  f"metadati dichiarati {row.metadata_kv_count}; ordine byte {row.byte_order}; "
                  "intestazione letta (contenuto e compatibilita' non verificati).")
        else:
            print(f"Ruth: GGUF {row.model_id}: stato {row.status}.")
    for reason in report.errors:
        print(f"Ruth: GGUF: problema {reason}.")
    print(f"Ruth: controllo GGUF: {report.status}; {report.count} intestazioni dichiarate.")
    if report.status in ("not_configured", "empty", "no_gguf"):
        print("Ruth: nessun file GGUF da ispezionare; nessun download.")
    return 1 if report.status in ("partial", "invalid") else 0


def main(argv: Sequence[str] = ()) -> int:
    """Run PY -> DEP -> DIAG -> HW -> NVIDIA -> CAT -> GGUF; targeted checks remain."""
    args = list(argv)
    root = Path(__file__).resolve().parent
    if not args:
        print("Ruth: sequenza automatica dei controlli integrati.")
        python = _check_python(root)
        if not python.interpreter_in_project_venv:
            print("Ruth: controllo dipendenze saltato: ambiente Python non conforme.")
            _diagnose(python, None)
            _check_hardware()
            _check_gpu()
            _check_catalog(root)
            _check_gguf(root)
            print("Ruth: sequenza controlli terminata con problemi.")
            return 1
        print("Ruth: dipendenze minime da controllare: pip (non elenco completo di Ruth).")
        code, dep = _check_dependencies(DEFAULT_CHECK_PACKAGES, root)
        _diagnose(python, dep)
        _check_hardware()
        _check_gpu()
        catalog_code = _check_catalog(root)
        gguf_code = _check_gguf(root)
        final_code = max(code, catalog_code, gguf_code)
        if final_code == 0:
            print("Ruth: sequenza controlli completata; compatibilita' runtime non verificata.")
        else:
            print("Ruth: sequenza controlli terminata con problemi.")
        return final_code

    if args == ["catalog"]:
        return _check_catalog(root)
    if args == ["gguf"]:
        return _check_gguf(root)

    if args[0] != "dependencies" or len(args) < 2:
        print("Ruth: uso: python main.py [catalog | gguf | dependencies PACCHETTO [PACCHETTO==VERSIONE ...]]")
        return 2
    return _check_dependencies(args[1:], root)[0]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
