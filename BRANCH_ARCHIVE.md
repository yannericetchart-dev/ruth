# Ruth: conservazione delle fonti storiche

Decisione: BRANCH-LIFECYCLE-001. Main esaminato: c1d3572627b0f544ae6ef54d05daca7ee2b62f62.
Stato: blocked_missing_authenticated_ref_mutations. Nessun tag creato.
I nomi proposti sotto non sono fonti gia' disponibili: i branch originali
devono restare finche' i tag remoti non siano stati creati e verificati.

| Branch originario | Tip verificato | Commit non contenuti in main | Tag proposto, non creato |
| --- | --- | --- | --- |
| `environment/python/main` | `b80520ad903ff18f93398725ea95cd91f51da5d8` | 4 | `archive/2026-10-17/environment/python/main` |
| `environment/python/work/entrypoint` | `b80520ad903ff18f93398725ea95cd91f51da5d8` | 4 | `archive/2026-10-17/environment/python/work/entrypoint` |
| `environment/python/work/inspect-venv` | `80218739327666b0250c3715ba9cfb6fe1f114d5` | 2 | `archive/2026-10-17/environment/python/work/inspect-venv` |
| `feat/gemma3-flash-kv-policy` | `dc6039be1f17e09c6ff407dbee1ee6783eecb853` | 311 | `archive/2026-10-17/feat/gemma3-flash-kv-policy` |
| `feat/gemma3-kv-matrix-18` | `d0ffdea9711bebd18da9b1e7866d3f9cd8c2f09a` | 341 | `archive/2026-10-17/feat/gemma3-kv-matrix-18` |
| `feat/gemma3-kv-precision-plans` | `5080cf0c74ed1b92c0cb4530895c9911e3e189b7` | 310 | `archive/2026-10-17/feat/gemma3-kv-precision-plans` |
| `feat/gemma3-native-context-probes` | `9ed7b187126dce38a3609c1e9e62bd6a17b786bd` | 330 | `archive/2026-10-17/feat/gemma3-native-context-probes` |
| `feat/gemma3-phase27-numeric-benchmark` | `c09432b09f5f65e8f8428835d4347311d86ae6af` | 354 | `archive/2026-10-17/feat/gemma3-phase27-numeric-benchmark` |
| `feat/gemma3-phase28-context-dispatch` | `4ceea508025650ed64b33190834d0b5c85f69db2` | 473 | `archive/2026-10-17/feat/gemma3-phase28-context-dispatch` |
| `feature/minimal-chat` | `954f408ece2d7595bcb67447fa277acb9eaffc73` | 459 | `archive/2026-10-17/feature/minimal-chat` |
| `model-profiling` | `52b67662dff9409864dfd5da62059a1dfd4532cb` | 309 | `archive/2026-10-17/model-profiling` |

La storia e' ancora conservata dai branch esistenti. Prima di ogni rimozione:
riconfermare tip, assenza di PR/incarichi/integrazioni pendenti, creare il tag
autenticato, verificarne la dereferenziazione al commit, aggiornare i
riferimenti di migrazione e solo dopo cancellare il ramo.
I due nomi che puntano allo stesso tip restano entrambi documentati.
Per la fonte PY-001 il commit immutabile resta
4ceea508025650ed64b33190834d0b5c85f69db2; il futuro tag proposto per la fonte
e' archive/2026-10-17/feat/gemma3-phase28-context-dispatch, non ancora creato.
La fonte conversazione resta feature/minimal-chat al tip documentato,
senza importarne codice o dati in main.

Gli archivi verificati saranno duraturi e immutabili. Non marcare created
o verified senza aver letto il ref remoto e l'eventuale tag annotato.
Il passaggio a un tag non riscrive i riferimenti nelle registrazioni
storiche: aggiorna le fonti operative mantenendo nome originario e SHA.
