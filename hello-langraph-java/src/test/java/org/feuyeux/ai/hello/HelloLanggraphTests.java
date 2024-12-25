package org.feuyeux.ai.hello;

import static org.feuyeux.ai.hello.repository.EmbeddingModelBuilder.buildEmbeddingModel;
import static org.feuyeux.ai.hello.service.LanggraphService.getTavilyApiKey;
import static org.feuyeux.ai.hello.service.LanggraphService.getZhipuAiKey;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.zhipu.ZhipuAiEmbeddingModel;
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.chroma.ChromaEmbeddingStore;
import java.time.Duration;
import java.util.List;
import java.util.stream.Collectors;
import lombok.extern.slf4j.Slf4j;
import org.bsc.langgraph4j.GraphRepresentation;
import org.feuyeux.ai.hello.fun.*;
import org.feuyeux.ai.hello.repository.HelloEmbeddingStore;
import org.feuyeux.ai.hello.util.DotEnvConfig;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
@Slf4j
class HelloLanggraphTests {

  private String question = "agent memory";
  @Autowired private HelloEmbeddingStore helloEmbeddingStore;

  @BeforeAll
  public static void beforeAll() throws Exception {
    DotEnvConfig.load();
  }

  @Test
  void test() {}

  @Test
  public void QuestionRewriterTest() {
    QuestionRewriter questionRewriter = QuestionRewriter.of(getZhipuAiKey());
    String result = questionRewriter.apply(question);
    log.info("question:{} QuestionRewriter result:{}", question, result);
    assertEquals("What is the role of memory in an agent's functioning?", result);
  }

  @Test
  public void RetrievalGraderTest() {
    RetrievalGrader grader = RetrievalGrader.of(getZhipuAiKey());
    EmbeddingStore<TextSegment> chroma = helloEmbeddingStore.buildEmbeddingStore();
    ZhipuAiEmbeddingModel embeddingModel = buildEmbeddingModel(getZhipuAiKey());
    EmbeddingSearchRequest query =
        EmbeddingSearchRequest.builder()
            .queryEmbedding(embeddingModel.embed(question).content())
            .maxResults(1)
            .minScore(0.0)
            .build();
    EmbeddingSearchResult<TextSegment> relevant = chroma.search(query);
    List<EmbeddingMatch<TextSegment>> matches = relevant.matches();
    assertEquals(1, matches.size());
    RetrievalGrader.Score answer =
        grader.apply(RetrievalGrader.Arguments.of(question, matches.getFirst().embedded().text()));
    assertEquals("no", answer.binaryScore);
  }

  @Test
  public void WebSearchTest() {
    WebSearchToolFn webSearchTool = WebSearchToolFn.of(getTavilyApiKey());
    List<Content> webSearchResults = webSearchTool.apply(question);
    String result =
        webSearchResults.stream()
            .map(content -> content.textSegment().text())
            .collect(Collectors.joining("\n"));
    assertNotNull(result);
    log.info("question:{} TavilyApi result:{}", question, result);
  }

  @Test
  public void questionRouterTest() {
    QuestionRouter qr = QuestionRouter.of(getZhipuAiKey());
    QuestionRouter.Type result = qr.apply("What are the stock options?");
    assertEquals(QuestionRouter.Type.web_search, result);
    result = qr.apply(question);
    assertEquals(QuestionRouter.Type.vectorstore, result);
  }

  @Test
  public void generationTest() {
    String question = "agent memory";
    EmbeddingSearchResult<TextSegment> relevantDocs = helloEmbeddingStore.search(question);
    List<String> docs =
        relevantDocs.matches().stream().map(m -> m.embedded().text()).collect(Collectors.toList());
    Generation qr = Generation.of(getZhipuAiKey());
    String result = qr.apply(question, docs);
    log.info("question:{} generation result:{}", question, result);
  }

  @Test
  public void getGraphTest() throws Exception {
    AdaptiveRag adaptiveRag =
        new AdaptiveRag(getZhipuAiKey(), getTavilyApiKey(), helloEmbeddingStore);
    org.bsc.langgraph4j.StateGraph<AdaptiveRag.State> graph = adaptiveRag.buildGraph();
    GraphRepresentation plantUml =
        graph.getGraph(GraphRepresentation.Type.PLANTUML, "Adaptive RAG");
    log.info("plantUml:{}", plantUml.getContent());
    GraphRepresentation mermaid = graph.getGraph(GraphRepresentation.Type.MERMAID, "Adaptive RAG");
    log.info("mermaid{}", mermaid.getContent());
  }

  // @Test
  public void connect() throws Exception {
    ChromaEmbeddingStore chroma =
        new ChromaEmbeddingStore(
            "http://localhost:8000", "rag-chroma", Duration.ofMinutes(2), true, true);

    ZhipuAiEmbeddingModel embeddingModel = buildEmbeddingModel(getZhipuAiKey());
    Embedding queryEmbedding =
        embeddingModel.embed("What are the types of agent memory?").content();

    EmbeddingSearchRequest query =
        EmbeddingSearchRequest.builder()
            .queryEmbedding(queryEmbedding)
            .maxResults(3)
            .minScore(0.0)
            .build();
    EmbeddingSearchResult<TextSegment> relevant = chroma.search(query);
    List<EmbeddingMatch<TextSegment>> matches = relevant.matches();
    assertEquals(3, matches.size());
    System.out.println(matches);
  }
}
