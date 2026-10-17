# GGUF-FIXTURE-SCOPE-001 (MIG-012)

Approved 17 ottobre 2026. Three disposable synthetic GGUF fixtures: scalar v2 little-endian, Q4_0 v3 big-endian, Q4_K v3 little-endian, declared only in a temporary CAT manifest. Read-only, bounded existing parsers; 12th Ruth default stage FIXTURE plus `gguf-fixture` targeted command. Negative malformed magic and truncated payload extent tests. Legacy INDEX and BLOCKS retain size_unverified. No real models, weights, downloads, network, inference or CUDA. Unit/CI and recursive maps required; after integration personal GGUF-FIXTURE-MANUAL-001 required (MIG-FLOW-002-R2). DATA-001 and REMOTE-001 unchanged.

Status: approved, candidate; integration and CI pending.

## MIG-012 integrazione tecnica verificata (17 ottobre 2026)

- PR #71 (GGUF), #72 (Formats), #73 (Models) e #74 (main) integrate, commit applicativo main `b057957b5d96b3c09ecde0b6c28ffc0e977c6942`.
- CI per tutte e quattro le promozioni success, inclusi test sintetici, punto d'ingresso e mappe. Dodicesima fase Ruth FIXTURE, dopo KBLOCKS, e comando mirato `gguf-fixture`.
- Stato: implementation complete, verification passed (fixture e CI), integration merged. **GGUF-FIXTURE-MANUAL-001 pending**: richiede collaudo personale del default Ruth e test mirati dopo aggiornamento del checkout personale.
- Non sono stati testati modelli reali o letti pesi. Nessun download, rete, CUDA, backend o inferenza. Nuovo perimetro bloccato fino a collaudo personale e nuova approvazione (MIG-FLOW-001/002). DATA-001 e REMOTE-001 invariati.
