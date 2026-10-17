# Ruth: audit e stato della pulizia dei branch

Snapshot riconfermato il 17 ottobre 2026 tramite GitHub.
Main di confronto: c1d3572627b0f544ae6ef54d05daca7ee2b62f62.
97 branch: main, 85 con tip contenuto nella storia di main e 11 non contenuti.
Controllata anche la seconda pagina dell'elenco: vuota.
Nessuna PR o issue aperta al controllo iniziale. Nessun tag remoto presente.
L'inventario preliminare fornito dall'utente coincide con nomi e SHA live;
la classificazione e' stata ricalcolata tramite confronto dei commit GitHub.

## BRANCH-CLEANUP-001: prerequisiti correnti (17 ottobre 2026)

Base: 5b77b08693a808c359b294bf1ab40f4fba8f5137.
Incarico completo BRANCH-ARCHIVE-001, evidenze attribuite e gate:
ORCHESTRATION.md, sezione "incarico archivi e gate di cancellazione".
Esecutore nominativo: chat Codex locale 01a14a37-aa01-70a2-b9cf-6702224b121f,
titolo "Ruth - Work Local - Orchestrator", ruolo limitato di esecutore.
ORCH-20261017-B resta unico titolare di documenti comuni, mappe, PR e merge.
Dopo merge dei prerequisiti con CI positiva: solo creazione/verifica degli
11 tag del registro, nessuna cancellazione; mutazioni tuttora non tentate.
Resoconto esecutore: gh autenticato, push=true, checkout main pulito alla
base e invariato; scrittura indicata dai controlli con accesso esteso.
Prova temporanea positiva storica, non ripetuta in questo incarico.
B non attribuisce a se' accesso o verifiche locali dell'esecutore.
Inventario riferito 50/49/0: 99 branch, 87 contained_candidate e 11
archive_required, main protetto. B ha riconfermato ref main, PR #105 merged,
99 branch con seconda pagina vuota, zero PR/issue aperte, zero tag e 11 tip.
61 NODE/checkpoint esaminati; nessun incarico concorrente pubblicato trovato.
MODEL-001 state=complete; Models implementation=partial, storia CAT preservata.
Per tutti gli 11 rami users_release unconfirmed: cancellazioni bloccate.
Zero PR head/base non prova assenza di utilizzatori o integrazioni pendenti.
Dopo esiti archivi, B verifica e aggiorna fonti operative via PR/CI/merge;
solo dopo risoluzione utilizzatori rilascia elenco esatto branch/SHA.
Nessun elenco eliminabile ora. Cancellazioni richiederanno condizione SHA.
Tag creati 0; branch cancellati 0; documenti/PR dell'esecutore 0.
Il ramo documentale di B aggiunge un temporaneo, pending_cleanup dopo merge.
Nessuna sync, nuova copia/ambiente, installazione, esecuzione Ruth o test
applicativo in questo incarico. MIG-016 chiusa; MIG-017 non approvata.
Le sezioni precedenti/successive datate della successione descrivono
gli strumenti e i blocchi storici, senza annullare questa assegnazione.

## Esito operativo storico alla base dell'audit

cleanup_status: blocked_missing_authenticated_ref_mutations.
deleted_branches: 0. created_archive_tags: 0.
Il connettore autenticato disponibile espone letture, commit, branch e PR,
ma non create-tag/create-ref generico o delete-ref. La CLI gh segnala
autenticazione non valida. Nessuna credenziale o dettaglio host e' riportato.
Non sono stati tentati aggiramenti, recuperi di token o modifiche ai login.

Le 85 righe contained_candidate sono candidati, non certificazioni di
assenza di utilizzatori. MODEL-001 resta active nei metadati Models:
la sua descrizione riguarda il catalogo gia' integrato e non approva nuove
funzioni, ma il suo stato va risolto con il responsabile prima di rimuovere
un branch usato dal nodo. Nessuna assegnazione concreta a una chat branch
e' attestata da questo inventario. Il successore deve verificare incarichi
e dipendenze effettivi prima di ogni cancellazione.

La PR di questo protocollo richiede un ulteriore ramo temporaneo
`work/ORCH-SUCCESSION-BRANCH-LIFECYCLE`, non incluso nelle 97 righe dello snapshot base.
Dopo il merge e in assenza di altri utilizzatori anch'esso e' candidato
alla rimozione, ma resta finche' manca una mutazione autenticata.
Numero finale di branch da rileggere live; non dichiarare la pulizia conclusa.

## Inventario riconfermato

| Branch | SHA | Classificazione | Commit non contenuti |
| --- | --- | --- | --- |
| `ci/python-isolation` | `7d1c668bf68a62829946ab787613801305249a6f` | contained_candidate | 0 |
| `docs/CAT-001-final-checkpoint` | `bbc6de5bbcf40f9b9052685b814f0f545a7d0ada` | contained_candidate | 0 |
| `docs/CAT-MANUAL-001-complete` | `863cf14cdf999c22e2cf1375cc161cb2acb1413c` | contained_candidate | 0 |
| `docs/DEP-001-final-checkpoint` | `3e1e023fcbc1087930d0b6b823d1dcc795d24dde` | contained_candidate | 0 |
| `docs/DIAG-001-final` | `5343b4e01d0a99badc3f79fd03861a4b5c9b064a` | contained_candidate | 0 |
| `docs/DIAG-MANUAL-001-complete` | `515bea063dba65436354fee3bf125b01d6ffb2cb` | contained_candidate | 0 |
| `docs/GGUF-001-final-checkpoint` | `b3e27ed715931c015af8a38470b4ea8beee84996` | contained_candidate | 0 |
| `docs/GGUF-BLOCK-001-final-checkpoint` | `091515bd20dea3de8e1e8e2f381cd588625dc2fc` | contained_candidate | 0 |
| `docs/GGUF-BLOCK-MANUAL-001-complete` | `774de775be7c1498fb000aecd8b2fff843a41d06` | contained_candidate | 0 |
| `docs/GGUF-FIXTURE-001-integrated` | `fe973443ac8bc89d2d9d0521e2f0bde3aeaf067f` | contained_candidate | 0 |
| `docs/GGUF-FIXTURE-MANUAL-001-closure` | `503bf0e42d473625a955bd4583e25e740b5672b4` | contained_candidate | 0 |
| `docs/GGUF-INDEX-001-final-checkpoint` | `751b2ed90dff7aafa0d502e6203ac81e0ca71883` | contained_candidate | 0 |
| `docs/GGUF-INDEX-MANUAL-001-complete` | `97e0710ac348cf047ff39ddfbdd23bb1dfeb893e` | contained_candidate | 0 |
| `docs/GGUF-KBLOCK-001-final-checkpoint` | `0e85f312e1853f245c99a69ec4db2d9c9f87b0f4` | contained_candidate | 0 |
| `docs/GGUF-KBLOCK-MANUAL-001-and-MIG-FLOW-002-r2` | `4b12f5f81afdc60d30e2e0427cb31ac61c6fd482` | contained_candidate | 0 |
| `docs/GGUF-MANUAL-001-complete` | `045a170a859a63e187fa5f291c46731e46a04a06` | contained_candidate | 0 |
| `docs/GGUF-META-001-final-checkpoint` | `e381a91df19dcf94120c0f8b4a7ae629677797ff` | contained_candidate | 0 |
| `docs/GGUF-META-MANUAL-001-complete` | `88bca19509cbae97dab9833f1f6776f2e72c2ae3` | contained_candidate | 0 |
| `docs/GGUF-STRUCT-STABILITY-001-integrated` | `fe56de41cef7193354a0edac7463f7f29e5ede36` | contained_candidate | 0 |
| `docs/GGUF-STRUCT-SUMMARY-001-integrated` | `5c1b2fb259d4496b04af2309d1e2c8889eda71d5` | contained_candidate | 0 |
| `docs/GGUF-STRUCT-SUMMARY-MANUAL-001-closure` | `cc7403486a24d669865cbcdb108605a3b6a84e06` | contained_candidate | 0 |
| `docs/GPU-001-final-checkpoint` | `25fe46f59be1a3f0d8f3ea3c52ad77ce6daa1083` | contained_candidate | 0 |
| `docs/GPU-MANUAL-001-complete` | `31384aed606ba620b0b428037590f2765f8cd797` | contained_candidate | 0 |
| `docs/HW-001-final-checkpoint` | `1779f7db5f1bdd525796a744c09255b865021d5c` | contained_candidate | 0 |
| `docs/HW-MANUAL-001-complete` | `33f6e75cf16595718328d357873e2d67fef443c9` | contained_candidate | 0 |
| `docs/MAIN-FLOW-001-final` | `dcc0908f7fa89d753da80c0d37c858dd8c98903a` | contained_candidate | 0 |
| `docs/MAIN-MANUAL-001-complete` | `0282c78bf3de8bf55d4c0344b75aea712a20f143` | contained_candidate | 0 |
| `docs/MIG-012-checkpoint-header` | `198ddc38c4b8771051660fe5283899dae5414d05` | contained_candidate | 0 |
| `docs/MIG-014-manual-closure` | `624efbe451d31ee84a8ef09284b64fe0591f5e0a` | contained_candidate | 0 |
| `docs/MIG-015-integrated` | `53a888e06c98eac2925c7e35b9d8667cf09110d5` | contained_candidate | 0 |
| `docs/MIG-015-manual-closure` | `48c45fb4e198e2d4ba4ae0ef6726a80e450c9cad` | contained_candidate | 0 |
| `docs/MIG-016-integrated` | `8db9dfae31fa19c795ddd24a5bfcf8adafa49e97` | contained_candidate | 0 |
| `docs/PY-001-final-checkpoint` | `9a9235c9f0ea8a0c2168fe0261175f23c1ac48ee` | contained_candidate | 0 |
| `docs/PY-MANUAL-001-confirmed` | `72f133aad595dcec9e92ff82e90622f6f8f80a48` | contained_candidate | 0 |
| `docs/PY-MANUAL-001-positive-run` | `b16b3843b323587b516aff77740ae4440826e809` | contained_candidate | 0 |
| `docs/REMOTE-001` | `a4e342f60c48af794cf9dfc179c8d27fda7a3517` | contained_candidate | 0 |
| `docs/SYNC-001-pre-resume-gate` | `1eb0c5101d0017739cff3b06666579b7531a7f04` | contained_candidate | 0 |
| `docs/SYNC-001` | `667dafbdc9862598ab5dca298667c50e96627a82` | contained_candidate | 0 |
| `environment/dependencies/main` | `b2bc0095407edf766c2563bbedc2e69889b074ce` | contained_candidate | 0 |
| `environment/dependencies/work/DEP-001` | `ea708c7471584e5972863a657b4b402d5339d0fe` | contained_candidate | 0 |
| `environment/diagnostics/main` | `3e8d1c19e35914ee69b287eb1a01d15677212543` | contained_candidate | 0 |
| `environment/diagnostics/work/DIAG-001` | `17a1cb65c98be40d8508b16f8ae69f0aa7fef3a3` | contained_candidate | 0 |
| `environment/main` | `3e8d1c19e35914ee69b287eb1a01d15677212543` | contained_candidate | 0 |
| `environment/python/main` | `b80520ad903ff18f93398725ea95cd91f51da5d8` | archive_required | 4 |
| `environment/python/work/entrypoint` | `b80520ad903ff18f93398725ea95cd91f51da5d8` | archive_required | 4 |
| `environment/python/work/inspect-venv` | `80218739327666b0250c3715ba9cfb6fe1f114d5` | archive_required | 2 |
| `environment/work/DEP-001-promotion` | `26a287f03bc83e4b025d94c548eaf3b086cd0012` | contained_candidate | 0 |
| `environment/work/PY-001-entrypoint` | `5fc816407f5c8873fd2ed39baa9d5293a9810c73` | contained_candidate | 0 |
| `feat/gemma3-flash-kv-policy` | `dc6039be1f17e09c6ff407dbee1ee6783eecb853` | archive_required | 311 |
| `feat/gemma3-kv-matrix-18` | `d0ffdea9711bebd18da9b1e7866d3f9cd8c2f09a` | archive_required | 341 |
| `feat/gemma3-kv-precision-plans` | `5080cf0c74ed1b92c0cb4530895c9911e3e189b7` | archive_required | 310 |
| `feat/gemma3-native-context-probes` | `9ed7b187126dce38a3609c1e9e62bd6a17b786bd` | archive_required | 330 |
| `feat/gemma3-phase27-numeric-benchmark` | `c09432b09f5f65e8f8428835d4347311d86ae6af` | archive_required | 354 |
| `feat/gemma3-phase28-context-dispatch` | `4ceea508025650ed64b33190834d0b5c85f69db2` | archive_required | 473 |
| `feature/minimal-chat` | `954f408ece2d7595bcb67447fa277acb9eaffc73` | archive_required | 459 |
| `hardware/cpu/main` | `ae492a1be3b12c0ffd6f9987d0d959aceb4a9ad4` | contained_candidate | 0 |
| `hardware/cpu/work/HW-001` | `6959fbd39a4d71665efd39110310b4df9927bd7c` | contained_candidate | 0 |
| `hardware/gpu/main` | `60e137a422403efa7d8d4005240abf317ce87e24` | contained_candidate | 0 |
| `hardware/gpu/nvidia/main` | `60e137a422403efa7d8d4005240abf317ce87e24` | contained_candidate | 0 |
| `hardware/gpu/nvidia/work/GPU-001` | `b355ed53bd182d8ebb2b981911885bc58719e2e6` | contained_candidate | 0 |
| `hardware/main` | `60e137a422403efa7d8d4005240abf317ce87e24` | contained_candidate | 0 |
| `main` | `c1d3572627b0f544ae6ef54d05daca7ee2b62f62` | permanent | 0 |
| `model-profiling` | `52b67662dff9409864dfd5da62059a1dfd4532cb` | archive_required | 309 |
| `models/catalog/main` | `90e57f106fb664b0257775260044125ea85228e2` | contained_candidate | 0 |
| `models/catalog/work/CAT-001` | `1d3f4cb1a25c706d3097a1a4455cd59150c47847` | contained_candidate | 0 |
| `models/formats/gguf/main` | `da9eba7392395ed05e81296aa4c595dc61d7b35e` | contained_candidate | 0 |
| `models/formats/gguf/work/GGUF-001` | `088cb7c4ac4d335d5b58d1beb47cd03218bdde7a` | contained_candidate | 0 |
| `models/formats/gguf/work/GGUF-BLOCK-001` | `a4913c38d96aba8b879c70ee4db383abc8037d23` | contained_candidate | 0 |
| `models/formats/gguf/work/GGUF-FIXTURE-001` | `c298582935e4abd800f0cd293219e1c0489cd2b1` | contained_candidate | 0 |
| `models/formats/gguf/work/GGUF-FIXTURE-SUMMARY-001` | `26e41fef60a0ea87cbeb655794bf7dfa6bcc0da3` | contained_candidate | 0 |
| `models/formats/gguf/work/GGUF-INDEX-001` | `469f5b4807059199ed855fafbeabb8e8e4654c49` | contained_candidate | 0 |
| `models/formats/gguf/work/GGUF-KBLOCK-001` | `61f00a4b347d229aaa64a77c9bd93fc6600af45d` | contained_candidate | 0 |
| `models/formats/gguf/work/GGUF-META-001` | `a7e663fd3789ff25495d6be42f8f28115e8f9b37` | contained_candidate | 0 |
| `models/formats/gguf/work/GGUF-MULTITENSOR-FIXTURE-001` | `477e12b225593f8370241b8b45c5816c9b9846d0` | contained_candidate | 0 |
| `models/formats/gguf/work/GGUF-STRUCT-STABILITY-001` | `0e3e8e4fe714ec57978c83921fb294c971728429` | contained_candidate | 0 |
| `models/formats/gguf/work/GGUF-STRUCT-SUMMARY-001` | `f6052d99c51c3f2c17b745babf741bc68d256c0d` | contained_candidate | 0 |
| `models/formats/main` | `f4218833623a45ad5574cf17ef993a455265141d` | contained_candidate | 0 |
| `models/formats/work/GGUF-FIXTURE-001` | `3d6a49b96a70a9258820e4878e7c6cbd8787eec0` | contained_candidate | 0 |
| `models/formats/work/GGUF-FIXTURE-SUMMARY-001` | `294361d0968d853c167ef8dbc222d82fa7a32438` | contained_candidate | 0 |
| `models/formats/work/GGUF-MULTITENSOR-FIXTURE-001` | `da9eba7392395ed05e81296aa4c595dc61d7b35e` | contained_candidate | 0 |
| `models/formats/work/GGUF-STRUCT-STABILITY-001` | `01a0e11e4cae56f87a0738b97d95282d88ed371b` | contained_candidate | 0 |
| `models/formats/work/GGUF-STRUCT-SUMMARY-001` | `77bbf827dc1f92ead4257126ad4dc3750a575d24` | contained_candidate | 0 |
| `models/main` | `d3a2826522654806ae60e9caae71382d9ab74ed2` | contained_candidate | 0 |
| `models/work/GGUF-FIXTURE-001` | `96b561245ddd1dec6bccd24175c1fdfc2b2612e6` | contained_candidate | 0 |
| `models/work/GGUF-FIXTURE-SUMMARY-001` | `388564de883873f6a583b93f17b5863b88fecba7` | contained_candidate | 0 |
| `models/work/GGUF-MULTITENSOR-FIXTURE-001` | `f4218833623a45ad5574cf17ef993a455265141d` | contained_candidate | 0 |
| `models/work/GGUF-STRUCT-STABILITY-001` | `f280fb9528c78f2cf750bc5e4cf14305cf13bd8e` | contained_candidate | 0 |
| `models/work/GGUF-STRUCT-SUMMARY-001` | `53bcf601a13cc3f4d5c624d69229358aa5fc57ae` | contained_candidate | 0 |
| `runtime/work/MAIN-FLOW-001` | `f624b09b53e4fd80d075a29bd45a2834773672df` | contained_candidate | 0 |
| `work/CONT-001-R2-ORCH-001` | `2afbfee834e2c11d4a71c6f8edffd938357a4212` | contained_candidate | 0 |
| `work/CONT-001-R2-close` | `ff3f0fb131e0e575dd7b527c93d709bd949a323e` | contained_candidate | 0 |
| `work/GGUF-FIXTURE-001` | `97b44491c7502d1123a3e856b57c85f30401ae58` | contained_candidate | 0 |
| `work/GGUF-FIXTURE-SUMMARY-001` | `0266a44aea496205dd55f648faf56b693656416d` | contained_candidate | 0 |
| `work/GGUF-MULTITENSOR-FIXTURE-001` | `d3a2826522654806ae60e9caae71382d9ab74ed2` | contained_candidate | 0 |
| `work/GGUF-STRUCT-STABILITY-001` | `b62e6fabaa2ffbb9bea06da6c976426321d8deb8` | contained_candidate | 0 |
| `work/GGUF-STRUCT-SUMMARY-001` | `c36957af18eac4d96ee77f937eb61a3930594af1` | contained_candidate | 0 |
| `work/MIG-016-delegated-close` | `a5f10c5ebf3824ed48beb3e3c249bc59d1d59204` | contained_candidate | 0 |

Per i rami archive_required vedere BRANCH_ARCHIVE.md: i tag sono soltanto
proposti. Nessun ramo e' stato eliminato e nessun NODE e' stato marcato
archived. Lo stato present dei rami tuttora esistenti resta corretto.
