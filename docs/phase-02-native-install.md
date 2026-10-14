# Fase 02 — Installazione CUDA nativa offline nella .venv

## Stato verificato sul PC

Il wheel nativo llama-cpp-python 0.3.36 CUDA sm_89 è già stato compilato e verificato quanto a contenuto delle librerie native. Non serve e non si deve ricompilare.

Percorso: .ruth-build/native-cuda-0.3.36-sm89/wheels/llama_cpp_python-0.3.36-py3-none-linux_x86_64.whl

SHA256 del file prodotto:
ad5057352154d90f95a473d01d039c975f1987cd9b25f8ad7a83cf6c7e80ee74

Il digest è registrato anche in locks/native-cuda-0.3.36-sm89-wheel.json.

## Comando 1: verifica senza modifiche

    ./ruth setup --plan-native-install

Verifica la .venv isolata Python 3.12, gli otto wheel Python già controllati,
il checksum del wheel CUDA prodotto, il file build-result.json, i
commit upstream, la presenza delle librerie native nel wheel, e l'assenza
di installazioni precedenti che possano sovrapporsi. Rifiuta symlink,
modifiche ai file e pin inattesi.

Lo stato atteso è ready_for_explicit_install_authorization. Non significa
ancora che l'offload sia stato realmente eseguito.

## Comando 2: installazione, soltanto con autorizzazione esplicita

La decisione di rinviare CVE-2025-69872 come TO DO non equivale ad accettare
automaticamente il rischio residuo di diskcache 5.6.3.

Il solo flag --install-native non basta: occorre esplicitare entrambe le
decisioni. Se si accetta questo rischio limitatamente al runtime Ruth
senza cache su disco, il comando è:

    ./ruth setup --install-native --accept-diskcache-risk

La CVE rimane aperta e l'opzione non equivale a dichiarare sicuro diskcache.
Ruth mantiene il divieto di LlamaDiskCache, cache pickle non attendibili e
plugin Python non attendibili nel processo del motore.

Procedura:
1. Verifica ancora gli SHA256 fissati di 8 wheel più il binding CUDA.
2. Installa offline in staging esterno alla .venv usando pip --target,
   --no-index e --no-deps, senza alcun nuovo download.
3. Avvia un nuovo interprete isolato della .venv (python -I -S) con percorso
   solo verso lo staging; verifica import di llama_cpp, versione 0.3.36 e
   llama_supports_gpu_offload() == True.
4. Solo dopo la prova positiva, trasferisce la directory nel contenitore
   .venv/ruth-native-cuda-0.3.36-sm89.
5. Attiva tramite un unico file .pth con percorso statico in
   .venv/lib/python3.12/site-packages/zz_ruth_native_cuda.pth;
   il file non contiene codice Python.
6. Ripete il controllo import/offload in un secondo nuovo processo; se
   fallisce, disattiva il .pth e ripristina la condizione precedente.

Un errore precedente all'attivazione lascia la .venv invariata. Percorsi
già esistenti o link simbolici vengono rifiutati, senza sovrascriverli.
Un arresto di corrente nel breve intervallo di attivazione non garantisce
il rollback automatico: un successivo avvio rifiuta la sovrascrittura
e richiede una riparazione manuale verificata.

Nessun cambio driver, nessun fallback CPU o wheel precompilato e nessuna
compilazione. Non avviare setup concorrenti nella stessa .venv.

## Verifica GPU e limiti

Un import con llama_supports_gpu_offload() == True prova la disponibilità
del backend CUDA nel binding, ma NON è ancora un'esecuzione GPU o inferenza
con modello GGUF sulla RTX 4050. I campi gpu_offload_executed e
inference_executed rimangono false. Anche ./ruth run rimane bloccato
finché un modello autorizzato e una vera inferenza non siano stati validati.

La CVE diskcache rimane TO DO non prioritario fino a release corretta
upstream e verifica di versione, digest, lock e test.
