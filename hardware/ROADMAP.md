# Hardware / hardware

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [HW](NODE.yaml). Ripresa: [START_HERE](../START_HERE.md).

Descrivere risorse senza confonderle con compatibilita di inferenza.

Sviluppo: **active**; codice: **complete**; maturita: experimental; verifica: passed; integrazione: merged.

Flag: in_scope. Blocchi: nessuno registrato.

Ambito verifica: GPU-001 NVIDIA: CI GitHub 38918374871, 38918650589, 38918759116, 38918867451 success (92 test sintetici, 258 mappe); promozioni PR #30 NVIDIA, #31 GPU, #32 hardware e #33 main. Driver, CUDA, AMD/Intel e compatibilita' inferenza non verificati. GPU-MANUAL-001 ancora pendente.

## Attivita locali

1. `HW-001` [complete] Contratti e rilevamento read-only CPU/RAM implementati, verificati in CI e integrati in main tramite PR #27.
2. `HW-SCOPE-001` [complete] Funzioni, ingressi, risultati, effetti, limiti e verifiche con CPU/RAM prima di GPU approvati dall'utente il 16 ottobre 2026.
3. `HW-MANUAL-001` [complete] Utente: copia main sincronizzata in fast-forward; commit 59b26fd6b427e6756ce0285fee60590cb077ebf1 verificato; avvio main.py completato con ordine PY->DEP->DIAG->HW e rilevamento CPU/RAM con stato available, 16 ottobre 2026 (MIG-FLOW-002).
4. `HW-PROMOTE-001` [complete] PR #26 promossa da hardware/cpu/main in hardware/main; merge d2ea16b14b89cefacea574f1e209fecec29f7f01.
5. `HW-PROMOTE-002` [complete] PR #27 promossa in main; merge 35a6ac79a1b1287090c1a3d1dc9208d4129a15ec.
6. `HW-GPU-001` [complete] Estensione NVIDIA GPU-001 read-only verificata e integrata in main (PR #33), con CPU/RAM preservate e collaudo personale GPU ancora pendente.

## Componenti (ordine locale, ID stabili)

1. [CPU: CPU](cpu/ROADMAP.md) — active; complete; passed.
2. [GPU: GPU](gpu/ROADMAP.md) — active; complete; passed.

Propagare impatti e blocchi al [padre](../ROADMAP.md) e proseguire fino alla radice.
