# HW-SCOPE-001: inventario CPU e RAM

Perimetro approvato esplicitamente dall'utente il 16 ottobre 2026
(MIG-FLOW-001); commit iniziale main:
`3e8d1c19e35914ee69b287eb1a01d15677212543`.

## Funzioni, ingressi, uscite

- `hardware.cpu.detect() -> tuple[DeviceInfo, ...]`: architettura
  `platform.machine()` e processori logici da `os.cpu_count()`.
  Entrambe le misure possono risultare `None`; non promettere core fisici.
- `hardware.resources(device='system') -> ResourceReport`: RAM fisica
  totale in **byte** da `os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')`
  quando supportato; altrimenti `unknown`. Il parametro accetta
  `system` oppure `DeviceInfo` di tipo CPU; un device GPU non e' supportato.
- `hardware.detect() -> HardwareReport`: aggrega una CPU e RAM totale
  con stati `available`, `partial` o `unknown`, senza certificare
  compatibilita' di inferenza. Contratti `DeviceInfo`,
  `ResourceReport`, `HardwareReport` immutabili e dati transitori
  mediante `to_dict()` ove previsto.
- `main.py` senza argomenti: PY -> DEP -> DIAG -> HW. L'inventario HW
  e' indipendente dall'isolamento della .venv e viene mostrato anche se
  PY/DEP non sono conformi. I sottocomandi mirati di DEP restano invariati.
  L'output hardware esplicita RAM/CPU osservabili e i valori sconosciuti.

## Controlli ed effetti

Sola lettura tramite libreria standard Python. Nessuna installazione,
download, benchmark, esplorazione GPU, invocazione di tool esterni,
driver, test backend, CUDA o inferenza. La RAM e' totale fisica, **non**
libera, disponibile, riservabile dal processo o limite container.
Su OS che non espone sysconf con tali chiavi RAM e' `unknown`.
Il successo dell'inventario non implica GPU o runtime disponibile.
Inventari e valori reali non vengono salvati su Git (DATA-001).

Test CI su fixture sintetiche: CPU nota/parziale/sconosciuta, conteggio
non valido, sysconf mancante/invalido/valori parziali, API non supportate,
report immutabile, ordinamento e blocchi del default Ruth e mappe.
Promozione remota: `hardware/cpu/work/HW-001` ->
`hardware/cpu/main` -> `hardware/main` -> `main`.
Poi sincronizzazione sicura e collaudo personale `HW-MANUAL-001`
(MIG-FLOW-002). La migrazione GPU e' separata.

Stato corrente: codice candidato implementato; verifica remota e
integrazione ancora pendenti.
