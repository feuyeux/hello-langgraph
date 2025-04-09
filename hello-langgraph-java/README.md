# Hello LangGraph for Java

- [bsorrentino/langgraph4j](https://github.com/bsorrentino/langgraph4j)

## Dev

### Dependency

```xml
<!-- https://mvnrepository.com/artifact/org.bsc.langgraph4j/langgraph4j-langchain4j -->
<dependency>
    <groupId>org.bsc.langgraph4j</groupId>
    <artifactId>langgraph4j-langchain4j</artifactId>
    <version>1.1.5</version>
</dependency>
```

## Test

### Run LangGraph Adaptive RAG Demo

```sh
sh test.sh testGraphRun
```


### Generate Adaptive RAG Graph Diagram

```sh
sh test.sh testGraphing
```

[`src/main/java/org/feuyeux/ai/hello/graph/AdaptiveRagGraph.java`](src/main/java/org/feuyeux/ai/hello/graph/AdaptiveRagGraph.java)

| mermaid                                                                                  | plantUml                                                                                    |
|------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| <img src="AdaptiveRAG-mermaid.svg" alt="Adaptive RAG mermaid svg" style="width:600px" /> | <img src="Adaptive RAG plantUml svg" alt="Adaptive_RAG-plantUml.svg" style="width:600px" /> |


### Debug Environment

```sh
sh test.sh testLog
```