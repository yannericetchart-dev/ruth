# Environment: checkpoint PY-001

Perimetro PY-SCOPE-001 approvato il 16 ottobre 2026; ispezione Python
e ingresso Ruth integrati in main e accettati personalmente dall'utente.

- PY-PROMOTE-001: PR #2 in environment/python/main.
- PY-PROMOTE-002: PR #3 in environment/main,
  merge fd61723874052053f38446ee82eca954870804e1.
- PY-PROMOTE-003: PR #8 in main,
  merge ce1b0bb47b4ae4ad0da40f3f83c964b66015680f.
- Verifiche GitHub: run 38885972247 e 38886317198, success.
- PY-MANUAL-001: **complete**, esecuzione dal punto d'ingresso
  `python3 main.py` conforme, revisione verificata dall'utente
  `ef4654af8ab106ef194af254d9f3884e5cabb6b0`, 16 ottobre 2026.

Il sottosistema environment resta partial/experimental: gli altri
componenti non sono certificati. La vecchia cartella locale con
modifiche resta intatta e fuori dalla sincronizzazione automatica.
La copia collaudata e' aggiornata almeno alla revisione provata,
ma per i successivi commit di sola documentazione e' necessaria
una sincronizzazione fast-forward separata (REMOTE-001).
Non includere dati di host o log in Git secondo DATA-001.

Prossimo perimetro funzionale da sottoporre all'utente;
nessun lavoro applicativo successivo e' autorizzato implicitamente.

## DEP-001: verifica dipendenze generiche

DEP-SCOPE-001 approvato il 16 ottobre 2026. Codice nel branch
`environment/dependencies/work/DEP-001`, non ancora verificato o promosso.
La diagnostica DIAG-001 non e' inclusa. Dopo i test, propagare le mappe,
integrare a environment/main e main, poi collaudare personalmente.

## Verifica e integrazione DEP-001 nei branch figli

- PR #12 -> environment/dependencies/main: merge `4af2050e5fc59477f9ab854173bf915b7933d186`.
- PR #13 -> environment/main: merge `c89c02ac0afa828b2c53ebd922a04331069ba527`.
- Verifiche GitHub Actions 38895348468 e 38895479549: success;
  14 test Python, 12 dipendenze, 13 root/CLI e 237 mappe coerenti.
- RESTA: integrazione a main e collaudo personale DEP-MANUAL-001.
