package org.feuyeux.ai.hello.repository;

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
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Repository;
import org.testcontainers.chromadb.ChromaDBContainer;

import java.util.List;

import static dev.langchain4j.internal.Utils.randomUUID;
import static org.feuyeux.ai.hello.repository.EmbeddingModelBuilder.buildEmbeddingModel;
import static org.feuyeux.ai.hello.service.LanggraphService.getZhipuAiKey;

@Slf4j
@Repository
public class HelloEmbeddingStore {
    // https://hub.docker.com/r/chromadb/chroma/tags
    String tag = "0.5.23";
    private ChromaDBContainer chromaDb = new ChromaDBContainer("chromadb/chroma:" + tag);
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
        chromaDb.withWorkingDirectory("/tmp/chroma").start();
        embeddingStore =
                ChromaEmbeddingStore.builder()
                        .baseUrl(chromaDb.getEndpoint())
                        .collectionName(randomUUID())
                        .logRequests(true)
                        .logResponses(true)
                        .build();
    }

    @PostConstruct
    public void destroy() {
        chromaDb.stop();
    }
}
