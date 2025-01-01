#!/bin/bash
cd "$(
    cd "$(dirname "$0")" >/dev/null 2>&1
    pwd -P
)/" || exit

source ./env
echo "TAVILY_API_KEY=$TAVILY_API_KEY"
echo "ZHIPUAI_API_KEY=$ZHIPUAI_API_KEY"

if [ "$(docker ps -q -f name=chromadb)" ]; then
  echo "ChromaDB container is already running."
  docker ps -f name=chromadb
else
  sh ./chroma_start.sh
fi

mvn spring-boot:run