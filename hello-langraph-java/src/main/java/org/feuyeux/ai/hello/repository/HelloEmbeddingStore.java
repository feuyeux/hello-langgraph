package org.feuyeux.ai.hello.repository;

import static dev.langchain4j.internal.Utils.randomUUID;
import static org.feuyeux.ai.hello.repository.EmbeddingModelBuilder.buildEmbeddingModel;
import static org.feuyeux.ai.hello.service.LanggraphService.getZhipuAiKey;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.UrlDocumentLoader;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentByParagraphSplitter;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.zhipu.ZhipuAiEmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.chroma.ChromaEmbeddingStore;
import jakarta.annotation.PostConstruct;
import java.util.List;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Repository;
import org.testcontainers.chromadb.ChromaDBContainer;

@Slf4j
@Repository
public class HelloEmbeddingStore {
  // https://hub.docker.com/r/chromadb/chroma/tags
  // docker run --rm --name chromadb -v ./chroma:/chroma/chroma -e IS_PERSISTENT=TRUE -e ANONYMIZED_TELEMETRY=TRUE -p 8000:8000 chromadb/chroma:0.5.23

  private ZhipuAiEmbeddingModel embeddingModel = buildEmbeddingModel(getZhipuAiKey());
  private EmbeddingStore<TextSegment> embeddingStore;

  public EmbeddingStore<TextSegment> buildEmbeddingStore() {
    List.of(
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/")
        .forEach(
            url -> {
              Document document = UrlDocumentLoader.load(url, new TextDocumentParser());
              int maxSegmentSize = 30;
              DocumentSplitter splitter = new DocumentByParagraphSplitter(maxSegmentSize, 0);
              List<TextSegment> segments = splitter.split(document);
              log.info("url:{} segments:{}", url, segments.size());
              segments.forEach(
                  segment -> {
                    Embedding embedding = embeddingModel.embed(segment).content();
                    embeddingStore.add(embedding, segment);
                  });
            });
    return embeddingStore;
  }

  public EmbeddingSearchResult<TextSegment> search(String query) {
    Embedding queryEmbedding = embeddingModel.embed(query).content();
    EmbeddingSearchRequest searchRequest =
        EmbeddingSearchRequest.builder()
            .queryEmbedding(queryEmbedding)
            .maxResults(1)
            .minScore(0.0)
            .build();
    return embeddingStore.search(searchRequest);
  }

  @PostConstruct
  public void init() {
    embeddingStore =
        ChromaEmbeddingStore.builder()
            .baseUrl("http://localhost:8000")
            .collectionName(randomUUID())
            .logRequests(true)
            .logResponses(true)
            .build();
  }
}
