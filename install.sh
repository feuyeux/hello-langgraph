#!/bin/sh

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd -P)
cd "$SCRIPT_DIR" || exit

case "$(uname -s)" in
    *Linux*|*Darwin*)
        python -m venv lg_env
        echo "install on WSL, Linux or macOS"
        . "$SCRIPT_DIR/lg_env/bin/activate"
        which python
        pip install --upgrade pip
        ;;
    *CYGWIN*|*MSYS*|*MINGW*)
        python -m venv graph_env
        echo "install on Cygwin or MSYS2"
        . "graph_env/Scripts/activate"
        python -m pip install --upgrade pip
        ;;
    *)
        echo "Unsupported OS"
        exit 1
        ;;
esac
pip install -r requirements.txt
