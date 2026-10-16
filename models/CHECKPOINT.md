# Models: checkpoint CAT-001

- CAT-SCOPE-001 approvato 16 ottobre 2026; solo catalogo locale
  read-only dichiarativo e opzionale, sesto passaggio di Ruth.
- models/catalog candidata, con test sintetici/CI e promozioni ancora
  da compiere. Il resto di Models e' ancora scaffold inattivo:
  GGUF, sorgenti/acquisizione Hugging Face, cataloghi remoti,
  verifica degli artefatti e inferenza esclusi.
- CAT-MANUAL-001 non svolto: dopo l'integrazione in main servira'
  conferma della revisione eseguita personalmente dall'utente.

## CAT-001 verificata e integrata (16 ottobre 2026)

- PR #36 foglia, #37 models/main e #38 main integrate. GitHub CI
  38928909391, 38929028829 e 38929118836 success (116 test sintetici,
  264 mappe). Si tratta soltanto del catalogo locale dichiarativo.
- Il dominio Models resta parziale: catalogo completo per il solo
  perimetro approvato, GGUF/download/Hugging Face non implementati.
- CAT-MANUAL-001 active: in attesa di avvio personale Ruth dalla
  revisione main sincronizzata. Nessun dato host salvato su Git.
