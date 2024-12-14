#!/bin/sh

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd -P)
cd "$SCRIPT_DIR" || exit

case "$(uname -s)" in
    *Linux*|*Darwin*)
        echo "run on WSL, Linux or macOS"
        . "$SCRIPT_DIR/lg_env/bin/activate"
        which python
        ;;
    *CYGWIN*|*MSYS*|*MINGW*)
        echo "run on Cygwin or MSYS2"
        . "graph_env/Scripts/activate"
        which python
        ;;
    *)
        echo "Unsupported OS"
        exit 1
        ;;
esac
python hello_reflexion.py
