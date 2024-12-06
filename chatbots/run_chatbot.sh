#!/bin/sh
# shellcheck disable=SC1091
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd -P)
cd "$SCRIPT_DIR" || exit

case "$(uname -s)" in
    *Linux*|*Darwin*)
        echo "run on WSL, Linux or macOS"
        . "../graph_env/bin/activate"
        ;;
    *CYGWIN*|*MSYS*|*MINGW*)
        echo "run on Cygwin or MSYS2"
        . "../graph_env/Scripts/activate"
        ;;
    *)
        echo "Unsupported OS"
        exit 1
        ;;
esac
which python
echo "python hello_chatbot.py"
python hello_chatbot.py
