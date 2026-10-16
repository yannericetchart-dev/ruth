# Hardware: checkpoint HW-001

**HW-SCOPE-001 implementato, verificato e integrato in main.**
Perimetro approvato il 16 ottobre 2026: CPU e RAM, non GPU.

- Contratti frozen `HardwareReport`, `ResourceReport`, `DeviceInfo`.
  Ingressi `hardware.detect()`, `hardware.resources('system')`,
  `hardware.cpu.detect()`; valori unknown quando non osservabili.
- Ingresso standard `main.py`: PY -> DEP -> DIAG -> HW, con inventario
  CPU/RAM eseguito anche quando PY non e' conforme.
- PR #25 CPU foglia, merge `bf8c72e4566c22b768b36b770c66bbd0f770dbaa`;
  PR #26 hardware/main, merge `d2ea16b14b89cefacea574f1e209fecec29f7f01`;
  PR #27 main, merge `35a6ac79a1b1287090c1a3d1dc9208d4129a15ec`.
- CI 38913317440, 38913438829, 38913533057 success:
  74 test sintetici, 249 mappe coerenti.
- HW-MANUAL-001: **active**, in attesa di sincronizzazione della
  copia main pulita e di esecuzione personale `.venv/bin/python main.py`.
- Nessuna certificazione GPU, backend, prestazioni o RAM libera.
  Nessun dato reale hardware o host su Git secondo DATA-001.
