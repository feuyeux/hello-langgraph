#!/bin/bash
cd "$(
    cd "$(dirname "$0")" >/dev/null 2>&1
    pwd -P
)/" || exit

#export JAVA_HOME=/usr/local/opt/openjdk/libexec/openjdk.jdk/Contents/Home
mvn clean install -DskipTests
