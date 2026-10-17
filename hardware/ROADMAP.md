# Hardware / hardware

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [HW](NODE.yaml). Ripresa: [START_HERE](../START_HERE.md).

Descrivere risorse senza confonderle con compatibilita di inferenza.

Sviluppo: **active**; codice: **complete**; maturita: experimental; verifica: passed; integrazione: merged.

Flag: in_scope, temporary_branch. Blocchi: nessuno registrato.

Ambito verifica: GPU-001 NVIDIA: CI runs 38918374871, 38918650589, 38918759116, 38918867451 success (92 test sintetici e 258 mappe); PR #30-#33 integrate. GPU-MANUAL-001 passato il 16 ottobre 2026 su c344c71013cfa05da4f49f3204900f4bdc89b8ea: cinque passaggi PY->DEP->DIAG->HW->NVIDIA, stato NVIDIA available e riepilogo positivo. Non certifica CUDA, driver, backend o inferenza; dati host esclusi.

## Attivita locali

1. `HW-001` [complete] Contratti e rilevamento read-only CPU/RAM implementati, verificati in CI e integrati in main tramite PR #27.
2. `HW-SCOPE-001` [complete] Funzioni, ingressi, risultati, effetti, limiti e verifiche con CPU/RAM prima di GPU approvati dall'utente il 16 ottobre 2026.
3. `HW-MANUAL-001` [complete] Utente: copia main sincronizzata in fast-forward; commit 59b26fd6b427e6756ce0285fee60590cb077ebf1 verificato; avvio main.py completato con ordine PY->DEP->DIAG->HW e rilevamento CPU/RAM con stato available, 16 ottobre 2026 (MIG-FLOW-002).
4. `HW-PROMOTE-001` [complete] PR #26 promossa da hardware/cpu/main in hardware/main; merge d2ea16b14b89cefacea574f1e209fecec29f7f01.
5. `HW-PROMOTE-002` [complete] PR #27 promossa in main; merge 35a6ac79a1b1287090c1a3d1dc9208d4129a15ec.
6. `HW-GPU-001` [complete] Estensione NVIDIA GPU-001 read-only verificata, integrata in main via PR #33 e collaudata personalmente nel flusso Ruth; CPU/RAM preservate.

## Componenti (ordine locale, ID stabili)

1. [CPU: CPU](cpu/ROADMAP.md) — active; complete; passed.
2. [GPU: GPU](gpu/ROADMAP.md) — active; complete; passed.

Propagare impatti e blocchi al [padre](../ROADMAP.md) e proseguire fino alla radice.
