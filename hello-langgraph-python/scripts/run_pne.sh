#!/bin/sh

# Get the project root directory (parent of scripts/)
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
SCRIPT_DIR="$PROJECT_ROOT/scripts"

cd "$PROJECT_ROOT" || exit

case "$(uname -s)" in
*Linux* | *Darwin*)
    echo "run on WSL, Linux or macOS"
    . "$PROJECT_ROOT/lg_env/bin/activate"
    which python
    ;;
*CYGWIN* | *MSYS* | *MINGW*)
    echo "run on Cygwin or MSYS2"
    . "$PROJECT_ROOT/graph_env/Scripts/activate"
    which python
    ;;
*)
    echo "Unsupported OS"
    exit 1
    ;;
esac
python scripts/hello_pne.py
