"""Official Ruth entrypoint: synthetic end-to-end orchestration tests."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import unittest
from unittest.mock import patch

import main
from environment.runtime.contracts import DependencyPlan, PackageState, PythonEnvironment
from hardware import DeviceInfo, ResourceReport, HardwareReport
from hardware.gpu import GPUDevice, GPUReport
from models.catalog import ModelCatalog, ModelRecord
from models.formats.gguf import GGUFReport, GGUFHeader, GGUFMetadata, GGUFMetadataReport, GGUFTensorIndex, GGUFTensorIndexReport


def python_report(root: str, good: bool, reasons: tuple[str, ...]) -> PythonEnvironment:
    return PythonEnvironment(
        project_root=root, interpreter="synthetic",
        python_version=(3, 12, 0), interpreter_in_project_venv=good,
        venv_status="ok" if good else "invalid_or_inactive",
        reasons=reasons,
    )


def dep_report(python: PythonEnvironment, status: str = "present_unverified") -> DependencyPlan:
    return DependencyPlan(python, (
        PackageState("pip", "pip", "24.0" if status == "present_unverified" else None, status),
    ), python.interpreter_in_project_venv and status == "present_unverified")


def hw_report():
    return HardwareReport((DeviceInfo("cpu", "synthetic", 4, "available"),),
                          ResourceReport(4096, "available"), "available")


def gpu_report(status="available"):
    if status == "available":
        return GPUReport((GPUDevice("NVIDIA", "synthetic", 2048, "available"),), status)
    return GPUReport((), status)


def model_report(status="not_configured"):
    entries = (ModelRecord("synthetic", "gguf", "synthetic.gguf", "missing"),) if status == "partial" else ()
    errors = ("invalid_schema",) if status == "invalid" else ()
    return ModelCatalog(status, entries, errors)


class EntrypointTests(unittest.TestCase):
    def setUp(self):
        self.gguf_patcher = patch("main.inspect_headers", return_value=GGUFReport("not_configured", ()))
        self.gguf_mock = self.gguf_patcher.start()
        self.addCleanup(self.gguf_patcher.stop)
        self.meta_patcher = patch("main.inspect_all_metadata", return_value=GGUFMetadataReport("not_configured", ()))
        self.meta_mock = self.meta_patcher.start()
        self.addCleanup(self.meta_patcher.stop)
        self.index_patcher = patch("main.inspect_all_tensor_indexes", return_value=GGUFTensorIndexReport("not_configured", ()))
        self.index_mock = self.index_patcher.start()
        self.addCleanup(self.index_patcher.stop)

    def test_ok_runs_python_then_dependencies_then_diagnostics(self):
        py = python_report("synthetic", True, ())
        out = StringIO()
        with (
            patch("main.inspect_environment", return_value=py) as inspect,
            patch("main.plan_dependencies", return_value=dep_report(py)) as plan,
            patch("main.detect_hardware", return_value=hw_report()) as hw,
            patch("main.detect_gpu", return_value=gpu_report()) as gpu,
            patch("main.load_model_catalog", return_value=model_report()) as cat,
            redirect_stdout(out),
        ):
            result = main.main()
        self.assertEqual(result, 0)
        inspect.assert_called_once_with(Path(main.__file__).resolve().parent)
        plan.assert_called_once_with(main.DEFAULT_CHECK_PACKAGES, project_root=Path(main.__file__).resolve().parent)
        printed = out.getvalue()
        self.assertLess(printed.index("avvio controllo ambiente Python"),
                        printed.index("avvio controllo dipendenze Python"))
        self.assertLess(printed.index("avvio controllo dipendenze Python"),
                        printed.index("avvio diagnostica ambiente"))
        self.assertIn("diagnostica: ok_unverified", printed)
        self.assertLess(printed.index("avvio diagnostica ambiente"), printed.index("avvio inventario hardware"))
        hw.assert_called_once_with()
        gpu.assert_called_once_with()
        cat.assert_called_once_with(project_root=Path(main.__file__).resolve().parent)
        self.gguf_mock.assert_called_once_with(project_root=Path(main.__file__).resolve().parent)
        self.meta_mock.assert_called_once_with(project_root=Path(main.__file__).resolve().parent)
        self.index_mock.assert_called_once_with(project_root=Path(main.__file__).resolve().parent)
        self.assertIn("RAM fisica totale (byte): 4096", printed)
        self.assertLess(printed.index("avvio inventario hardware"), printed.index("avvio inventario GPU NVIDIA"))
        self.assertIn("GPU NVIDIA: 1 rilevate; stato available", printed)
        self.assertLess(printed.index("avvio inventario GPU NVIDIA"), printed.index("avvio controllo catalogo modelli locale"))
        self.assertIn("catalogo modelli: not_configured", printed)
        self.assertLess(printed.index("avvio controllo catalogo modelli locale"), printed.index("avvio controllo intestazioni GGUF dichiarate"))
        self.assertIn("controllo GGUF: not_configured", printed)
        self.assertLess(printed.index("avvio controllo intestazioni GGUF dichiarate"), printed.index("avvio controllo metadati GGUF dichiarati"))
        self.assertIn("metadati GGUF: not_configured", printed)
        self.assertLess(printed.index("avvio controllo metadati GGUF dichiarati"), printed.index("avvio controllo indici tensori GGUF dichiarati"))
        self.assertIn("indici GGUF: not_configured", printed)
        self.assertIn("sequenza controlli completata", printed)

    def test_invalid_python_still_diagnosed_but_packages_skipped(self):
        py = python_report("synthetic", False, ("configuration_missing", "prefix_mismatch"))
        out = StringIO()
        with (
            patch("main.inspect_environment", return_value=py),
            patch("main.plan_dependencies") as plan,
            patch("main.detect_hardware", return_value=hw_report()) as hw,
            patch("main.detect_gpu", return_value=gpu_report()) as gpu,
            patch("main.load_model_catalog", return_value=model_report()) as cat,
            redirect_stdout(out),
        ):
            result = main.main()
        self.assertEqual(result, 1)
        plan.assert_not_called()
        text = out.getvalue()
        self.assertIn("controllo dipendenze saltato", text)
        self.assertLess(text.index("controllo dipendenze saltato"),
                        text.index("avvio diagnostica ambiente"))
        self.assertIn("venv_configuration_missing", text)
        self.assertIn("venv_prefix_mismatch", text)
        self.assertIn("catalogo modelli: not_configured", text)
        self.assertLess(text.index("avvio inventario hardware"), text.index("avvio inventario GPU NVIDIA"))
        self.assertLess(text.index("avvio diagnostica ambiente"), text.index("avvio inventario hardware"))
        hw.assert_called_once_with()
        gpu.assert_called_once_with()
        cat.assert_called_once_with(project_root=Path(main.__file__).resolve().parent)

    def test_missing_dependency_explained(self):
        py = python_report("synthetic", True, ())
        out = StringIO()
        with (
            patch("main.inspect_environment", return_value=py),
            patch("main.plan_dependencies", return_value=dep_report(py, "missing")),
            patch("main.detect_hardware", return_value=hw_report()) as hw,
            patch("main.detect_gpu", return_value=gpu_report()) as gpu,
            patch("main.load_model_catalog", return_value=model_report()) as cat,
            redirect_stdout(out),
        ):
            result = main.main()
        self.assertEqual(result, 1)
        self.assertIn("pacchetto pip: missing", out.getvalue())
        self.assertIn("package_missing", out.getvalue())
        self.assertIn("sequenza controlli terminata con problemi", out.getvalue())
        hw.assert_called_once_with()
        gpu.assert_called_once_with()
        cat.assert_called_once_with(project_root=Path(main.__file__).resolve().parent)

    def test_failed_metadata_lookup_explained(self):
        py = python_report("synthetic", True, ())
        out = StringIO()
        with (
            patch("main.inspect_environment", return_value=py),
            patch("main.plan_dependencies", return_value=dep_report(py, "inspection_failed")),
            patch("main.detect_hardware", return_value=hw_report()) as hw,
            patch("main.detect_gpu", return_value=gpu_report()) as gpu,
            patch("main.load_model_catalog", return_value=model_report()) as cat,
            redirect_stdout(out),
        ):
            result = main.main()
        self.assertEqual(result, 1)
        self.assertIn("diagnostica: inspection_incomplete", out.getvalue())
        hw.assert_called_once_with()
        gpu.assert_called_once_with()
        cat.assert_called_once_with(project_root=Path(main.__file__).resolve().parent)

    def test_missing_nvidia_is_not_failure_of_other_steps(self):
        py = python_report("synthetic", True, ())
        out = StringIO()
        with (
            patch("main.inspect_environment", return_value=py),
            patch("main.plan_dependencies", return_value=dep_report(py)),
            patch("main.detect_hardware", return_value=hw_report()),
            patch("main.detect_gpu", return_value=gpu_report("unavailable")),
            patch("main.load_model_catalog", return_value=model_report()) as cat,
            redirect_stdout(out),
        ):
            code = main.main()
        self.assertEqual(code, 0)
        self.assertIn("GPU NVIDIA: 0 rilevate; stato unavailable", out.getvalue())
        self.assertIn("sequenza controlli completata", out.getvalue())

    def test_targeted_dependencies_do_not_invoke_gpu(self):
        py = python_report("synthetic", True, ())
        out = StringIO()
        with (
            patch("main.plan_dependencies", return_value=dep_report(py)),
            patch("main.detect_gpu") as gpu,
            redirect_stdout(out),
        ):
            code = main.main(("dependencies", "pip"))
        self.assertEqual(code, 0)
        gpu.assert_not_called()


    def test_invalid_catalog_only_fails_last_step(self):
        py = python_report("synthetic", True, ())
        out = StringIO()
        with (
            patch("main.inspect_environment", return_value=py),
            patch("main.plan_dependencies", return_value=dep_report(py)),
            patch("main.detect_hardware", return_value=hw_report()),
            patch("main.detect_gpu", return_value=gpu_report()),
            patch("main.load_model_catalog", return_value=model_report("invalid")),
            redirect_stdout(out),
        ):
            code = main.main()
        self.assertEqual(code, 1)
        self.assertIn("catalogo: problema invalid_schema", out.getvalue())
        self.assertIn("sequenza controlli terminata con problemi", out.getvalue())

    def test_missing_catalog_model_is_reported_and_nonzero(self):
        py = python_report("synthetic", True, ())
        out = StringIO()
        with (
            patch("main.inspect_environment", return_value=py),
            patch("main.plan_dependencies", return_value=dep_report(py)),
            patch("main.detect_hardware", return_value=hw_report()),
            patch("main.detect_gpu", return_value=gpu_report()),
            patch("main.load_model_catalog", return_value=model_report("partial")),
            redirect_stdout(out),
        ):
            code = main.main()
        self.assertEqual(code, 1)
        self.assertIn("modello synthetic", out.getvalue())
        self.assertIn("stato missing", out.getvalue())

    def test_targeted_catalog_checks_only_catalog(self):
        out = StringIO()
        with (
            patch("main.load_model_catalog", return_value=model_report()) as cat,
            patch("main.detect_gpu") as gpu,
            patch("main.inspect_environment") as python,
            redirect_stdout(out),
        ):
            code = main.main(("catalog",))
        self.assertEqual(code, 0)
        cat.assert_called_once_with(project_root=Path(main.__file__).resolve().parent)
        gpu.assert_not_called()
        python.assert_not_called()
        self.gguf_mock.assert_not_called()
        self.meta_mock.assert_not_called()
        self.index_mock.assert_not_called()


class GGUFEntrypointTests(unittest.TestCase):
    def test_targeted_gguf_is_header_only(self):
        header = GGUFHeader("synthetic", "header_read", 3, 2, 1, "little")
        with patch("main.inspect_headers", return_value=GGUFReport("inspected", (header,))) as inspect, patch("main.inspect_environment") as py:
            out = StringIO()
            with redirect_stdout(out):
                status = main.main(("gguf",))
        self.assertEqual(status, 0)
        self.assertIn("versione 3", out.getvalue())
        inspect.assert_called_once_with(project_root=Path(main.__file__).resolve().parent)
        py.assert_not_called()

    def test_invalid_gguf_is_reported_and_failed(self):
        record = GGUFHeader("synthetic", "invalid_magic")
        with patch("main.inspect_headers", return_value=GGUFReport("partial", (record,))):
            out = StringIO()
            with redirect_stdout(out):
                code = main.main(("gguf",))
        self.assertEqual(code, 1)
        self.assertIn("stato invalid_magic", out.getvalue())


class MetadataEntrypointTests(unittest.TestCase):
    def test_targeted_metadata_only(self):
        row = GGUFMetadata("synthetic", "metadata_read", "llama", 15, 4096)
        with patch("main.inspect_all_metadata", return_value=GGUFMetadataReport("inspected", (row,))) as inspect, patch("main.inspect_environment") as py:
            out = StringIO()
            with redirect_stdout(out):
                result = main.main(("gguf-metadata",))
        self.assertEqual(result, 0)
        self.assertIn("architettura dichiarata llama", out.getvalue())
        self.assertIn("contesto dichiarato 4096", out.getvalue())
        inspect.assert_called_once_with(project_root=Path(main.__file__).resolve().parent)
        py.assert_not_called()

    def test_targeted_malformed_metadata_nonzero(self):
        row = GGUFMetadata("synthetic", "truncated_metadata")
        with patch("main.inspect_all_metadata", return_value=GGUFMetadataReport("partial", (row,))):
            out = StringIO()
            with redirect_stdout(out):
                status = main.main(("gguf-metadata",))
        self.assertEqual(status, 1)
        self.assertIn("stato truncated_metadata", out.getvalue())


class IndexEntrypointTests(unittest.TestCase):
    def test_targeted_index_check(self):
        record = GGUFTensorIndex("synthetic", "index_checked", 1)
        with patch("main.inspect_all_tensor_indexes", return_value=GGUFTensorIndexReport("inspected", (record,))) as inspect, patch("main.inspect_environment") as py:
            out = StringIO()
            with redirect_stdout(out):
                code = main.main(("gguf-index",))
        self.assertEqual(code, 0)
        self.assertIn("1 descrittori coerenti", out.getvalue())
        inspect.assert_called_once_with(project_root=Path(main.__file__).resolve().parent)
        py.assert_not_called()

    def test_unknown_tensor_type_is_not_misreported_as_verified(self):
        record = GGUFTensorIndex("synthetic", "size_unverified", 2, 1)
        with patch("main.inspect_all_tensor_indexes", return_value=GGUFTensorIndexReport("partial", (record,))):
            out = StringIO()
            with redirect_stdout(out):
                code = main.main(("gguf-index",))
        self.assertEqual(code, 1)
        self.assertIn("stato size_unverified", out.getvalue())


if __name__ == "__main__":
    unittest.main()
