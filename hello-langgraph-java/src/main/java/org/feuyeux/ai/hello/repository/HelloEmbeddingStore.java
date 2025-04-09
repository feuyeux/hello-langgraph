package org.feuyeux.ai.hello.repository;

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
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.chroma.ChromaEmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import jakarta.annotation.PostConstruct;
import java.util.List;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Repository;

@Slf4j
@Repository
public class HelloEmbeddingStore {
  private ZhipuAiEmbeddingModel embeddingModel;
  private EmbeddingStore<TextSegment> embeddingStore;

  public EmbeddingStore<TextSegment> buildEmbeddingStore() {
    EmbeddingSearchResult<TextSegment> relevant = search("agent memory");
    List<EmbeddingMatch<TextSegment>> matches = relevant.matches();
    if (matches.size() < 1) {
      log.info("Building EmbeddingStore...");
      List.of(
              "https://lilianweng.github.io/posts/2023-06-23-agent/",
              "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
              "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/")
          .forEach(
              url -> {
                try {
                  Document document = UrlDocumentLoader.load(url, new TextDocumentParser());
                  int maxSegmentSize = 1024;
                  DocumentSplitter splitter = new DocumentByParagraphSplitter(maxSegmentSize, 8);
                  List<TextSegment> segments = splitter.split(document);
                  log.info("url:{} segments:{}", url, segments.size());
                  segments.forEach(
                      segment -> {
                        Embedding embedding = embeddingModel.embed(segment).content();
                        embeddingStore.add(embedding, segment);
                      });
                } catch (Exception e) {
                  log.error("Error loading document from {}: {}", url, e.getMessage());
                }
              });
    }
    log.info("EmbeddingStore is ready.");
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
    try {
      log.info("Attempting to initialize ChromaEmbeddingStore...");
      embeddingStore =
          ChromaEmbeddingStore.builder()
              .baseUrl("http://localhost:8000")
              .collectionName("hello-embedding-store")
              .logRequests(true)
              .logResponses(true)
              .build();
      log.info("ChromaEmbeddingStore initialized successfully");
    } catch (Exception e) {
      log.warn(
          "Failed to initialize ChromaEmbeddingStore: {}. Using InMemoryEmbeddingStore instead.",
          e.getMessage());
      embeddingStore = new InMemoryEmbeddingStore<>();
    }

    embeddingModel = buildEmbeddingModel(getZhipuAiKey());
    try {
      buildEmbeddingStore();
    } catch (Exception e) {
      log.error("Error building embedding store: {}", e.getMessage());
    }
  }
}
