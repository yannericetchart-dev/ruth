#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3 non trovato. Installa Python 3 e riprova." >&2
    exit 1
fi

if ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
    echo "Ruth richiede Python 3.10 o superiore." >&2
    exit 1
fi

if [[ ! -x .venv/bin/python ]] || ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
    echo "Creazione ambiente virtuale .venv..."
    if ! python3 -m venv .venv; then
        echo "Impossibile creare il venv. Su Ubuntu installa python3-venv" >&2
        echo "(sudo apt install python3-venv), poi riesegui questo script." >&2
        exit 1
    fi
fi

echo "Installazione dipendenze nell'ambiente virtuale..."
.venv/bin/python -m pip install -r requirements.txt

echo
echo "Ambiente pronto. Avvia Ruth con:"
echo "  .venv/bin/python main.py"
