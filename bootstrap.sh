#!/bin/sh
# Ruth bootstrap: works before Python or the project .venv exists.
# This script NEVER installs system packages, downloads models, or changes drivers.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
VENV="$ROOT/.venv"

usage() {
    printf '%s\n' "Usage: sh bootstrap.sh --plan | --check | --create-venv" >&2
    exit 2
}

python_advice() {
    distro="your GNU/Linux distribution"
    family=""
    if [ -r /etc/os-release ]; then
        # os-release is system metadata, not arbitrary user-supplied input.
        # Avoid sourcing it as executable shell code.
        distro=$(sed -n 's/^ID=//p' /etc/os-release | head -n 1 | tr -d '"')
        family=$(sed -n 's/^ID_LIKE=//p' /etc/os-release | head -n 1 | tr -d '"')
    fi
    printf '%s\n' "Python 3.10+ and the venv module are required to create Ruth's .venv."
    case " $distro $family " in
        *" ubuntu "*|*" debian "*) printf '%s\n' "Debian/Ubuntu: sudo apt update && sudo apt install python3 python3-venv" ;;
        *" fedora "*|*" rhel "*) printf '%s\n' "Fedora/RHEL: sudo dnf install python3 (install the distribution's Python venv support if separated)" ;;
        *" arch "*) printf '%s\n' "Arch: sudo pacman -S python" ;;
        *" opensuse "*|*" suse "*) printf '%s\n' "openSUSE: use zypper to install a supported python3 and python3-venv package if provided" ;;
        *) printf '%s\n' "Install a supported Python >= 3.10 with venv using the packages for $distro." ;;
    esac
    printf '%s\n' "System packages require the user's approval. A venv does not install Python itself."
}

check_platform() {
    if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
        printf '%s\n' "ERROR: Ruth targets GNU/Linux x86_64 only." >&2
        return 1
    fi
}

check_python() {
    if ! command -v python3 >/dev/null 2>&1; then
        printf '%s\n' "ERROR: python3 is missing." >&2
        python_advice
        return 1
    fi
    if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
        printf '%s\n' "ERROR: Python 3.10+ is required." >&2
        python_advice
        return 1
    fi
}

check_venv() {
    if [ ! -f "$VENV/pyvenv.cfg" ] || [ ! -x "$VENV/bin/python" ]; then
        printf '%s\n' "ERROR: Ruth .venv does not exist or is incomplete: $VENV" >&2
        printf '%s\n' "Run: sh bootstrap.sh --create-venv (explicitly creates it)." >&2
        return 1
    fi
    if grep -Eiq '^[[:space:]]*include-system-site-packages[[:space:]]*=[[:space:]]*true' "$VENV/pyvenv.cfg"; then
        printf '%s\n' "ERROR: .venv exposes global site-packages: unsupported for Ruth." >&2
        printf '%s\n' "Create a fresh isolated virtualenv after approving cleanup of the existing one." >&2
        return 1
    fi
    if ! "$VENV/bin/python" -c 'import pathlib,sys; sys.exit(0 if pathlib.Path(sys.prefix).resolve() == pathlib.Path(sys.argv[1]).resolve() else 1)' "$VENV"; then
        printf '%s\n' "ERROR: Ruth .venv is invalid or belongs to another path." >&2
        printf '%s\n' "No existing environment will be overwritten automatically." >&2
        return 1
    fi
}

if [ "$#" -ne 1 ]; then usage; fi
case "$1" in
    --plan)
        check_platform || exit 1
        printf '%s\n' "Ruth setup plan — no files will be changed."
        python_advice
        printf '%s\n' "1. Verify Python 3.10+ on the host (needed to create .venv)."
        printf '%s\n' "2. Create ./ .venv with: sh bootstrap.sh --create-venv"
        printf '%s\n' "3. Install APPROVED, VERSION-PINNED Python dependencies inside .venv (not defined yet)."
        printf '%s\n' "4. Verify NVIDIA driver, CUDA-capable llama.cpp binding and native inference separately."
        printf '%s\n' "No system or Python dependencies are installed by this command."
        ;;
    --check)
        check_platform || exit 1
        check_python || exit 1
        check_venv || exit 1
        printf '%s\n' "OK: Python and the project-local Ruth .venv are available."
        printf '%s\n' "Next: .venv/bin/python -m ruth_runtime.dependencies"
        ;;
    --create-venv)
        check_platform || exit 1
        check_python || exit 1
        if [ -e "$VENV" ]; then
            check_venv || exit 1
            printf '%s\n' "OK: Existing Ruth .venv retained; no packages installed."
        else
            # Only explicit --create-venv authorizes creating files.
            if ! python3 -m venv "$VENV"; then
                printf '%s\n' "ERROR: creating .venv failed (venv/ensurepip may be unavailable)." >&2
                python_advice
                printf '%s\n' "Inspect incomplete .venv manually; the script will not delete or replace it." >&2
                exit 1
            fi
            check_venv || exit 1
            printf '%s\n' "OK: Created $VENV"
        fi
        printf '%s\n' "No model, GPU library, or Python dependency installed."
        ;;
    *) usage ;;
esac
