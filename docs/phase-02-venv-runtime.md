# Ruth — ambiente virtuale, dipendenze e avvio

## Stato verificato sul PC (14 ottobre 2026)

- Python 3.12.3 nella `.venv` isolata, NVIDIA driver 595.99.02,
  CUDA Toolkit 13.2, GPU RTX 4050 Laptop `sm_89`.
- 87 test passati nel check precedente a questo cambiamento.
- Il kernel CUDA nativo è stato compilato, eseguito e validato.
- **Non** sono stati installati `llama-cpp-python` né i pacchetti runtime necessari;
  non esiste un modello LLM verificato, né un adapter operativo.
- I nuovi test del launcher non sono ancora stati eseguiti sul PC dell'utente.

## Ingresso unico di Ruth

`./ruth` esegue sempre `.venv/bin/python`: non cerca pacchetti Python
globali e non dipende dall'attivazione della shell. Se la `.venv` manca,
solo il comando esplicito `./ruth setup` può crearla tramite il bootstrap
già verificato. Non installa pacchetti di sistema.

| Comando | Comportamento |
| --- | --- |
| `./ruth doctor` | Legge Python, `.venv`, pacchetti installati; non scrive. |
| `./ruth plan` | Aggiunge la diagnostica CUDA/Git/sorgenti pinned; non scrive. |
| `./ruth setup` | Valida la `.venv` e si ferma se il download non è autorizzato. |
| `./ruth setup --allow-downloads` | Autorizza e acquisisce automaticamente soltanto i sorgenti pinned, riutilizzando quelli già verificati. |
| `./ruth setup --allow-python-dependency-downloads` | Se il checkout è già verificato, autorizza soltanto **pip download** di wheel binari runtime e build in staging. Genera il manifest candidato con SHA256 e si ferma prima dell'installazione. |
| `./ruth setup --allow-downloads --allow-python-dependency-downloads` | Acquisisce i sorgenti se mancanti e poi risolve i wheel candidati automaticamente, con due autorizzazioni dichiarate nella stessa invocazione. |
| `./ruth run` | **Rifiuta** l'inferenza finché binding/adapter e modello non saranno realmente installati e verificati. |

## Diagnosi delle dipendenze

Il modulo `ruth_runtime.venv_packages` legge i metadati nella `.venv`
**solo se** verifica `sys.prefix`, `pyvenv.cfg`, isolamento da site-packages
dell'host e Python >=3.10. Altrimenti dichiara `blocked` senza accedere
alla lista dei pacchetti dell'interprete corrente.

Requisiti runtime upstream di `llama-cpp-python v0.3.36`:
`typing-extensions>=4.5.0`, `numpy>=1.20.0`,
`diskcache>=5.6.1`, `jinja2>=2.11.3`, transitiva `MarkupSafe>=2.0`;
versione del binding Ruth `llama-cpp-python==0.3.36`.
Requisito di build iniziale: `scikit-build-core>=0.9.2`, incluse
tutte le transitive che saranno rilevate e bloccate.

Gli stati `missing`, `incompatible` e `present_unverified` distinguono
pacchetti assenti, binding di versione sbagliata e metadati presenti senza
un lock validato. Un pacchetto «presente» **non** significa compatibile con
tutti i vincoli: finché non viene verificato il lock completo, i flag
`runtime_dependencies_verified`, `python_packages_installable`,
`backend_gpu_verified` e `ruth_runtime_ready` restano `false`.

## Generazione automatica del lock candidato — in attesa di test

`./ruth setup --allow-python-dependency-downloads` utilizza **esclusivamente**
l'interprete `.venv/bin/python -m pip --isolated download` per ottenere i wheel
binari da `https://pypi.org/simple/`, comprese le dipendenze transitive.
Il comando **non installa**, non compila e non scarica il wheel CPU/GPU
precompilato di `llama-cpp-python`: quest'ultimo rimane destinato alla
compilazione CUDA locale dai sorgenti pinned.

Un'area di staging privata nella directory ignorata `.ruth-wheelhouse/`
contiene i wheel, le versioni e gli SHA256 calcolati. Dopo i controlli
sul formato e sui metadati, promuove il candidato in
`.ruth-wheelhouse/python-dependencies-candidate/` con
`candidate-lock.json`. Qualsiasi candidato preesistente è preservato
senza sovrascrittura. Fallimenti e download incompleti non vengono promossi.

**Gli SHA256 calcolati sui file scaricati non costituiscono da soli una
validazione della provenienza PyPI, della chiusura del grafo delle
dipendenze o della sicurezza.** Il manifest ha `approved=false` e
`dependency_lock_ready=false` finché una verifica indipendente dei digest,
dei metadati e dei vincoli, oltre alla policy di sicurezza, non sarà
implementata ed eseguita. Questo è un passaggio verso il lockfile,
non un'installazione già utilizzabile.

`diskcache<=5.6.3` ha CVE-2025-69872 senza release upstream corretta
conosciuta: Ruth mantiene il divieto di `LlamaDiskCache` e dei dati di cache
su disco non attendibili, ma deve ancora testare l'enforcement nell'adapter.
Non promuovere automaticamente il candidato ad installabile.

## Installazione automatica: gate da completare

Non inventiamo hash approvati. Dopo il lock candidato restano la verifica
indipendente dei digest/artefatti, la coerenza di tutto il grafo e la
revisione di sicurezza. Solo allora lo setup potrà essere autorizzato
a installare le dipendenze e a compilare CUDA.

Solo **dopo** questo gate, il comando `setup` potrà installare automaticamente
nella `.venv` i wheel corretti e compilare il binding CUDA dalla sorgente,
in modo ripetibile, senza `sudo pip`, senza downgrade silenziosi e senza
fallback CPU. Una compilazione fallita non autorizza il wheel precompilato:
il suo impiego richiede una scelta separata.

L'avvio di Ruth (`run`) richiederà poi probe GPU, offload e inferenza nativa
con un modello verificato. Finché manca tale adapter, `run` restituisce
un errore esplicito, non un messaggio di successo fittizio.
