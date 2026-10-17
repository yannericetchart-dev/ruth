# GGUF-MULTITENSOR-FIXTURE-SCOPE-001 — MIG-016

Approvata espressamente dall'utente il 17 ottobre 2026; base canonica main `3cd6cad3e431f5ad1ab8dc698aee96330ff79ed1`.

- Estendere FIXTURE con **quattro** file sintetici temporanei (i precedenti tre piu' `multi`) in catalogo CAT isolato in TemporaryDirectory. Il quarto GGUF v3 little-endian dichiara **tre tensori** con tipi scalar F32, Q4_0 e Q4_K e intervalli distinti. Pesi assenti: i byte placeholder sono soltanto zeri sintetici e non vengono letti dai controlli.
- Eseguire CAT -> GGUF -> META -> INDEX -> BLOCKS -> KBLOCKS -> SUMMARY sul catalogo sintetico. Attesi INDEX `partial`, BLOCKS `partial`, KBLOCKS `inspected`, SUMMARY `structural_checked` per tutti e quattro i modelli e `tensor_count=3` per `multi`. Nessun `size_unverified` residuo dopo KBLOCKS quando tutti i tipi sono noti.
- Verifiche mirate in `test_multitensor` (11 test): intervalli validi e descrittori fuori ordine, endian big e allineamento personalizzato, offset non allineato, sovrapposizioni, offset fuori file, estensione troncata, offset duplicato, tipo GGML sconosciuto, dimensione blocco non multipla. Mantenere test sintetici MIG-012/013/014/015 aggiornati a quattro fixture.
- Gli esiti negativi devono essere `invalid` con codice portabile esistente o `size_unverified` per tipi sconosciuti. Niente falsa certificazione `structural_checked`. Nessun cambiamento al parser bounded approvato o alla semantica di validazione: collaudo dei controlli esistenti, non introduzione di nuove categorie di quantizzazione.
- Conservare le **13 fasi** di `main.py`, FIXTURE continua a precedere SUMMARY del catalogo personale opzionale, e il sottocomando mirato `gguf-fixture` resta disponibile.
- Seguire REMOTE-001: PR e CI per GGUF -> Formats -> Models -> main, mappe ricorsive e checkpoint; poi test personale di Ruth default e `python -m unittest models.formats.gguf.tests.test_multitensor -v`. MIG-FLOW-002-R2: evidenza positiva completa implica accettazione senza altra domanda.
- DATA-001: nessun log, host inventory, percorsi personali, dati reali, credenziali o benchmark in Git. Nessun modello reale, download, lettura dei pesi, CUDA, rete o inferenza. Kaggle inattivo.

Stato iniziale: **approved**; implementation candidate; CI, integration and manual acceptance pending.
