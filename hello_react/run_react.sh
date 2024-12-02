#!/bin/bash
# shellcheck disable=SC1091
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd -P)
cd "$SCRIPT_DIR" || exit

if [[ "$(uname -s)" == *"Linux"* ]] || [[ "$(uname -s)" == *"Darwin"* ]]; then
    echo "run on WSL, Linux or macOS"
    source "../graph_env/bin/activate"
elif [[ "$(uname -s)" == *"CYGWIN"* ]] || [[ "$(uname -s)" == *"MSYS"* ]] || [[ "$(uname -s)" == *"MINGW"* ]]; then
    echo "run on Cygwin or MSYS2"
    . "../graph_env/Scripts/activate"
else
    echo "Unsupported OS"
    exit 1
fi
python hello_react.py
