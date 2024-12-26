#!/bin/bash

# https://hub.docker.com/r/chromadb/chroma/tags
docker run -d --rm --name chromadb -v "/d/garden/chromadb":/chroma/chroma \
-e IS_PERSISTENT=TRUE \
-e ANONYMIZED_TELEMETRY=TRUE \
-p 8000:8000 \
chromadb/chroma:0.5.23