# Models: checkpoint CAT-001

- CAT-SCOPE-001 approvato 16 ottobre 2026; solo catalogo locale
  read-only dichiarativo e opzionale, sesto passaggio di Ruth.
- models/catalog implementata, verificata e integrata in main tramite
  PR #36-#38; collaudo manuale CAT-MANUAL-001 superato. Il resto di Models e' ancora scaffold inattivo:
  GGUF, sorgenti/acquisizione Hugging Face, cataloghi remoti,
  verifica degli artefatti e inferenza esclusi.
- CAT-MANUAL-001 completato positivamente dall'utente il 16 ottobre 2026
  sul commit `90e57f106fb664b0257775260044125ea85228e2`.

## CAT-001 verificata e integrata (16 ottobre 2026)

- PR #36 foglia, #37 models/main e #38 main integrate. GitHub CI
  38928909391, 38929028829 e 38929118836 success (116 test sintetici,
  264 mappe). Si tratta soltanto del catalogo locale dichiarativo.
- Il dominio Models resta parziale: catalogo completo per il solo
  perimetro approvato, GGUF/download/Hugging Face non implementati.
- CAT-MANUAL-001 complete: conferma personale successiva alla CI
  e all'integrazione, registrata senza dati host su Git.

## Accettazione manuale CAT-001 (16 ottobre 2026)

- Conferma positiva dell'utente: Ruth eseguita dall'ingresso ufficiale
  sul commit `90e57f106fb664b0257775260044125ea85228e2` (sei controlli in sequenza).
- Catalogo non configurato: `not_configured` valido; non e' richiesta
  la creazione di un manifest per il collaudo.
- Prossima migrazione Models non autorizzata: attendere descrizione
  funzionale e nuova approvazione. DATA-001 e REMOTE-001 invariati.

## GGUF-SCOPE-001 avviato (16 ottobre 2026)

- Dopo CAT-MANUAL-001 e PR #40, perimetro GGUF header-only approvato.
- Lettore in `models/formats/gguf` e settimo passaggio root candidati;
  nessuna integrazione GGUF in main finche' non si concludono promozioni.
- Le restanti porzioni Models (GGUF payload, acquisizione, source Hugging Face)
  rimangono inattive; GGUF-MANUAL-001 dopo merge.

## GGUF-001 verificata e integrata (16 ottobre 2026)

- Reader e orchestrazione approvati (GGUF-SCOPE-001), test sintetici e
  CI passati nei quattro livelli; PR #41-#44 integrate fino a main
  al commit `15b106082c4dee37622c61a8afefd248a57464a0`.
- Il dominio Models resta partial: lettura di intestazione GGUF completata
  soltanto per il perimetro autorizzato, resto skeleton/inattivo.
- GGUF-MANUAL-001 pending; il collaudo personale avverra' dal flusso
  completo di Ruth dopo sincronizzazione sicura del checkout.
  Nessuna migrazione successiva prima della conferma.
- Dati macchina/manifest e pesi del modello rimangono esclusi da Git.
