#!/bin/bash
cd "$(
    cd "$(dirname "$0")" >/dev/null 2>&1
    pwd -P
)/" || exit

source ./env
echo "TAVILY_API_KEY=$TAVILY_API_KEY"
echo "ZHIPUAI_API_KEY=$ZHIPUAI_API_KEY"



mvn clean test -Dtest=org.feuyeux.ai.hello.StructedOutputTests#${1:-testOutput}

