#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3 non trovato." >&2
    exit 1
fi

if ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
    echo "Ruth richiede Python 3.10 o superiore." >&2
    exit 1
fi

if [[ ! -x .venv/bin/python ]] || ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
    echo "Creazione ambiente virtuale .venv..."
    if ! python3 -m venv .venv; then
        echo "Installa python3-venv (sudo apt install python3-venv) e riprova." >&2
        exit 1
    fi
fi

echo "Controllo GPU NVIDIA e VRAM..."
python3 - <<'PY'
from hardware import GPUManager

gpu = GPUManager.select("nvidia")
if gpu is None:
    found = ", ".join(item["vendor"] for item in GPUManager.detect()) or "nessuna"
    raise SystemExit(
        f"GPU NVIDIA con driver e VRAM accessibili non rilevata (rilevate: {found}). "
        "Controlla che 'nvidia-smi' funzioni. Per ora Ruth supporta solo NVIDIA CUDA."
    )
print(f"{gpu['name']}: {gpu['vram_total_mb']} MiB VRAM ({gpu['vram_free_mb']} MiB liberi)")
PY

# Non basta che CUDA funzioni sull'host: il binding dentro il venv
# deve essere stato compilato con supporto all'offload GPU.
if .venv/bin/python -c 'from llama_cpp.llama_cpp import llama_supports_gpu_offload; assert llama_supports_gpu_offload()' >/dev/null 2>&1; then
    echo "llama-cpp-python: offload GPU già disponibile nel venv."
else
    if command -v nvcc >/dev/null 2>&1; then
        CUDA_COMPILER="$(command -v nvcc)"
    elif [[ -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ]]; then
        CUDA_COMPILER="${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"
    else
        echo "CUDA Toolkit (nvcc) non trovato. Serve per compilare llama-cpp-python con CUDA." >&2
        echo "Installa un CUDA Toolkit compatibile col driver NVIDIA o imposta CUDA_HOME." >&2
        exit 1
    fi

    echo "Compilazione di llama-cpp-python con CUDA nel venv..."
    CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_COMPILER=${CUDA_COMPILER}" \
    FORCE_CMAKE=1 \
    .venv/bin/python -m pip install \
        --force-reinstall --no-cache-dir --no-binary=llama-cpp-python \
        "llama-cpp-python>=0.3.0"
fi

echo "Installazione delle dipendenze..."
.venv/bin/python -m pip install -r requirements.txt

if ! .venv/bin/python -c 'from llama_cpp.llama_cpp import llama_supports_gpu_offload; assert llama_supports_gpu_offload()' >/dev/null 2>&1; then
    echo "ERRORE: llama-cpp-python nel venv non supporta l'offload GPU." >&2
    exit 1
fi

echo "Configurazione accesso Hugging Face..."
# Chiede il token personale solo a chi esegue il setup da un terminale.
# Non usare read -s o echo del segreto: getpass() mantiene il token nascosto,
# huggingface_hub lo salva nella cache personale e non nelle credenziali Git.
# In CI, cron o installazioni headless, l'autenticazione viene saltata.
if ! .venv/bin/python -m model.hf_auth; then
    echo "Attenzione: autenticazione Hugging Face non configurata." >&2
    echo "Riprova con: .venv/bin/python -m model.hf_auth --force" >&2
    echo "I download dei modelli pubblici restano comunque disponibili." >&2
fi

echo "Configurazione accesso Kaggle..."
# Anche Kaggle usa un token personale facoltativo. Nessun token negli
# argomenti, nell'output, in Git o in variabili esportate dal setup.
# Se c'e' un token gia' configurato, non vengono chieste credenziali.
if ! .venv/bin/python -m model.kaggle_auth; then
    echo "Attenzione: autenticazione Kaggle non configurata." >&2
    echo "Riprova con: .venv/bin/python -m model.kaggle_auth --force" >&2
    echo "I modelli pubblici possono comunque essere accessibili." >&2
fi

echo "CUDA disponibile nel venv. Avvia Ruth con:"
echo "  .venv/bin/python main.py"
