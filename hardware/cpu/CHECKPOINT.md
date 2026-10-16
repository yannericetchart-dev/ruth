# CPU: checkpoint CPU-001

CPU-001 **implementata, verificata e integrata** in main.
Provider read-only stdlib: architettura piattaforma e numero logico
tramite `platform.machine` e `os.cpu_count` (possono essere unknown).
PR #25 foglia in hardware/cpu/main, #26 in hardware/main e #27 in main.
CI 38913317440, 38913438829, 38913533057 success, incluse 8 prove
CPU sintetiche e 12 hardware/RAM.
Collaudo HW-MANUAL-001 sulla macchina dell'utente ancora in attesa.
Non include core fisici certi, GPU, prestazioni o driver.
DATA-001: nessun inventario reale versionato.
