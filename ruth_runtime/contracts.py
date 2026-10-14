"""Interfacce *proposte* per Ruth. Nessun adattatore concreto in questa fase.

Le interfacce sono dichiarazioni strutturali (typing.Protocol). I provider
possono usare API native distinte, ma l'orchestrazione usa solo questi contratti.
Le firme relative alla generazione verranno definite nella fase runtime.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Protocol, Sequence


class ComponentKind(str, Enum):
    CPU_VENDOR = "cpu_vendor"
    GPU_VENDOR = "gpu_vendor"
    GPU_BACKEND = "gpu_backend"
    CPU_BACKEND = "cpu_backend"  # Integrazioni future: nessun backend CPU è target operativo.
    NATIVE_LIBRARY = "native_library"  # Librerie usate, non backend selezionabili.
    INFERENCE_ENGINE = "inference_engine"
    INITIALIZATION = "initialization"


class DevelopmentState(str, Enum):
    TARGET = "target"           # Da implementare e provare per primo.
    TO_DO = "TO DO"             # Da registrare, ma non eseguibile.
    IMPLEMENTED = "implemented" # Codice presente, non prova nativa sufficiente.
    VERIFIED = "verified"       # Verifica nativa legata a evidenze concrete.


@dataclass(frozen=True)
class Capability:
    kind: ComponentKind
    identifier: str
    state: DevelopmentState
    note: str = ""


@dataclass(frozen=True)
class DeviceInfo:
    """Inventario fisico senza assumere capacità backend specifiche."""

    identifier: str
    kind: str  # "cpu" | "gpu"; enum/rilevamento nella fase 01.
    vendor: str
    name: str
    total_memory_bytes: int | None = None
    available_memory_bytes: int | None = None
    # Optional hardware identifiers/telemetry: NOT backend compatibility evidence.
    pci_address: str | None = None
    uuid: str | None = None
    driver: str | None = None


@dataclass(frozen=True)
class StackSelection:
    """Stack di inferenza: librerie native come GGML CPU NON sono backend."""

    device_kind: str
    device_vendor: str
    backend: str
    engine: str
    initialization: str


@dataclass(frozen=True)
class StackAssessment:
    selection: StackSelection
    state: DevelopmentState
    reason: str

    @property
    def runnable(self) -> bool:
        return self.state == DevelopmentState.VERIFIED


class UnsupportedStackError(ValueError):
    """Combinazione non registrata: mai fare fallback nascosti."""


class HardwareProvider(Protocol):
    """Adapter vendor-specifico dietro una API di inventario comune."""

    def discover(self) -> Sequence[DeviceInfo]: ...


class ExecutionBackend(Protocol):
    """Un backend deve dichiarare le capacità effettive del dispositivo."""

    def probe(self, device: DeviceInfo) -> Mapping[str, object]: ...


class InferenceEngine(Protocol):
    """Motore logico: distinto dal backend e dalla modalità di avvio."""

    def accepts(self, backend: str, initialization: str) -> bool: ...


class InferenceSession(Protocol):
    """Porta runtime preliminare. Generazione/streaming: fase 14."""

    def close(self) -> None: ...


class InitializationAdapter(Protocol):
    """Factory che applica una modalità di inizializzazione specifica."""

    def load(
        self,
        model_path: Path,
        selection: StackSelection,
        options: Mapping[str, object],
    ) -> InferenceSession: ...
