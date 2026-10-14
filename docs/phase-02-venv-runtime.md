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
| `./ruth setup --verify-candidate` | Verifica il candidato esistente: ricalcola gli hash, confronta digest e dimensioni con i metadati ufficiali PyPI e usa `pip --dry-run --no-index` per valutare chiusura e compatibilità senza installare. L'opzione autorizza la sola consultazione dei metadati PyPI. |
| `./ruth setup --review-security` | Controllo offline del codice upstream, wheel `diskcache` e 12 test dell'adapter CUDA; non installa e non può approvare da solo il rischio residuo. |
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

## Verifica del candidato e blocco sicurezza

Dopo il download già effettuato è sufficiente il comando **unico**:

```bash
./ruth setup --verify-candidate
```

Richiede connessione per consultare **solo i metadati JSON di PyPI**.
Non riscarica wheel o sorgenti; non installa, non compila e non aggiorna il driver.
L'audit controlla gli SHA-256 dei file locali, il loro `METADATA`, il nome,
la versione, la dimensione e l'eventuale stato yanked sul JSON PyPI;
quindi esegue `pip --isolated install --dry-run --ignore-installed --no-index
--only-binary=:all:` contro i soli wheel in staging, imponendo versioni
esatte per tutte le distribuzioni. Il report deve includere esattamente
gli otto package selezionati, senza mancanti o aggiunte.

Il confronto con i checksum di PyPI **non verifica di per sé attestazioni
crittografiche del publisher**, né assicura assenza di vulnerabilità. Se
tutti i controlli tecnici passano, il report restituisce
`status=security_review_required`, con `lock_approved=false`,
`security_review_passed=false` e `installation_started=false`.
Qualsiasi mismatch blocca e non modifica gli artefatti già scaricati.

Nota `scikit-build-core 1.1.1`: upstream ha rimosso da 1.0.0 l'extra
vuoto `[pyproject]` ancora richiesto dai sorgenti `llama-cpp-python
v0.3.36`. I moderni installer solitamente emettono un warning senza
invalidare per questo la build; la successiva compilazione CUDA
sarà comunque una prova obbligatoria separata.

`diskcache 5.6.3` rimane vulnerabile a CVE-2025-69872 e il blocco
di sicurezza resta aperto: prima di installare il binding occorre
implementare e testare l'adapter senza `LlamaDiskCache`, senza accesso
a cache serializzate non attendibili, e definire una policy esplicita
per il rischio residuo.

## Adapter CUDA senza cache su disco (contratto preliminare)

L'API reale di `llama-cpp-python v0.3.36` è stata controllata sul commit
fissato: `Llama.__init__` crea `self.cache = None`;
`Llama.set_cache(...)` può abilitarla esplicitamente;
`LlamaDiskCache.__init__` costruisce `diskcache.Cache(cache_dir)`.

Il nuovo `ruth_runtime.cuda_adapter.CudaLlamaAdapter` usa
**esclusivamente** lo stack `gpu/nvidia/cuda/llama.cpp/llama_cpp_python`,
rifiuta configurazioni di cache, directory, backend CPU e GPU layers
manuali; richiede versione esatta del binding `0.3.36`,
`llama_supports_gpu_offload() = true`, e crea `Llama` imponendo
`n_gpu_layers=-1`. La sessione pubblica espone soltanto `generate`,
`cache_disabled` e `close`, non `set_cache` o il motore nativo.
Prima della generazione verifica che `Llama.cache is None`; un oggetto
con cache inattesa viene chiuso e rifiutato.

I test dell'adapter utilizzano un binding **simulato** e verificano
esplicitamente che non siano create cache dal percorso gestito da Ruth.
La prova non dimostra che librerie terze siano prive di code path
vulnerabili, né che i pesi vengano realmente elaborati dalla GPU.
Questo adapter non è ancora collegato al launcher `run` e non è stato
eseguito con un modello GGUF reale.

**CVE-2025-69872 rimane aperta**: la mitigazione riduce la superficie
di attacco controllata da Ruth ma non corregge `diskcache` stessa.
Il manifest resta `approved=false` e l'installazione è bloccata
fino a revisione del rischio residuo, gestione sicurezza del filesystem
ed evidenza nativa su un ambiente dedicato.

## Gate di sicurezza applicativa: `--review-security`

Dopo aver acquisito i sorgenti pinned e aver verificato gli 8 wheel
su PyPI, la verifica **offline** dell'adapter e del rischio cache si avvia con:

```bash
./ruth setup --review-security
```

Questo comando controlla automaticamente:
- **source pin**: checkout verificato di `llama-cpp-python v0.3.36` e submodule
  `llama.cpp`, ai commit già fissati;
- **contratto upstream tramite AST**: `Llama.__init__` inizializza
  `self.cache = None` ed espone `set_cache`; `LlamaDiskCache` esiste
  e crea effettivamente `diskcache.Cache` (superficie di rischio nota);
- **artefatto locale**: candidato `diskcache 5.6.3` e SHA-256 invariati;
- **test del percorso Ruth**: esegue automaticamente i test di
  `tests/test_phase02_cuda_adapter.py` e richiede esito `OK`
  con almeno 12 test eseguiti.

Un esito tecnico positivo produce `controls_verified_residual_risk`
e `mitigations_validated=true`. Significa che il percorso **gestito da
Ruth** non abilita la cache su disco nelle condizioni simulate e l'API
upstream corrisponde alle assunzioni; non significa che il pacchetto
sia corretto o che sia possibile eseguire Python non attendibile
nello stesso processo.

La vulnerabilità `CVE-2025-69872` richiede che un aggressore
possa modificare i file di una cache successivamente deserializzata.
Ruth non deve creare `LlamaDiskCache`, caricare cache serializzate,
accettare percorsi di cache esterni o esporre `set_cache` nella sua
API di produzione. Nessuna di queste mitigazioni rende
`diskcache` un pacchetto sicuro contro usi da altro codice Python:
**`security_review_passed=false`, `lock_approved=false` e
`installation_authorized=false` restano veri anche quando i
controlli sono superati**. Questo gate non importa `llama_cpp`,
non installa pacchetti e non compila.

Per sbloccare l'installazione resta una decisione esplicita sul rischio
residuo, una revisione del confine di esecuzione (nessun plugin Python
non attendibile nello stesso processo) e infine la build nativa
con test di offload CUDA e inferenza GGUF reale. Evitare qualunque
autorizzazione implicita nascosta nello stato di `setup`.

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
