package org.feuyeux.ai.hello;

import static org.bsc.langgraph4j.GraphRepresentation.Type.MERMAID;
import static org.bsc.langgraph4j.GraphRepresentation.Type.PLANTUML;
import static org.feuyeux.ai.hello.service.LanggraphService.getTavilyApiKey;
import static org.feuyeux.ai.hello.service.LanggraphService.getZhipuAiKey;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Collectors;
import lombok.extern.slf4j.Slf4j;
import org.bsc.langgraph4j.GraphRepresentation;
import org.bsc.langgraph4j.StateGraph;
import org.feuyeux.ai.hello.fun.*;
import org.feuyeux.ai.hello.graph.AdaptiveRagGraph;
import org.feuyeux.ai.hello.repository.HelloEmbeddingStore;
import org.feuyeux.ai.hello.service.LanggraphService;
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
  @Autowired private LanggraphService langgraphService;

  @BeforeAll
  public static void beforeAll() throws Exception {
    // Disable any proxy settings that might interfere with tests
    disableProxy();
    DotEnvConfig.load();
  }

  private static void disableProxy() {
    System.clearProperty("http.proxyHost");
    System.clearProperty("http.proxyPort");
    System.clearProperty("https.proxyHost");
    System.clearProperty("https.proxyPort");
  }

  private static void useProxy() {
    String port = "50864";
    System.setProperty("http.proxyHost", "127.0.0.1");
    System.setProperty("http.proxyPort", port);
    System.setProperty("https.proxyHost", "127.0.0.1");
    System.setProperty("https.proxyPort", port);
  }

  @Test
  public void testLog() {
    log.error("error");
  }

  @Test
  public void testGraphRun() {
    try {
      String result = langgraphService.generate(question);
      log.info("[Question]:\n{} \n[Graph result]:{}", question, result);
    } catch (Exception e) {
      log.error("", e);
    }
  }

  // ----

  @Test
  public void testQuestionRouter() {
    QuestionRouterEdgeFn qr = QuestionRouterEdgeFn.of(getZhipuAiKey());
    QuestionRouterEdgeFn.Type result = qr.apply("What are the stock options?");
    assertEquals(QuestionRouterEdgeFn.Type.web_search, result);
    result = qr.apply(question);
    assertEquals(QuestionRouterEdgeFn.Type.vectorstore, result);
  }

  @Test
  public void testWebSearch() {
    WebSearchNodeFn webSearchTool = WebSearchNodeFn.of(getTavilyApiKey());
    List<Content> webSearchResults =
        webSearchTool.apply("Who is the first president of the China?");
    String result =
        webSearchResults.stream()
            .map(content -> content.textSegment().text())
            .collect(Collectors.joining("\n"));
    assertNotNull(result);
    log.info("[Question]:\n{}\n[TavilyApi result]:\n{}", question, result);
  }

  @Test
  public void testRetrievalGrader() {
    RetrievalGraderNodeFn grader = RetrievalGraderNodeFn.of(getZhipuAiKey());
    // retrieve
    EmbeddingSearchResult<TextSegment> relevant = helloEmbeddingStore.search(question);
    List<EmbeddingMatch<TextSegment>> matches = relevant.matches();
    assertEquals(1, matches.size());
    String document = matches.getFirst().embedded().text();
    // grade
    RetrievalGraderNodeFn.Arguments arguments =
        RetrievalGraderNodeFn.Arguments.of(question, document);
    RetrievalGraderNodeFn.Score answer = grader.apply(arguments);
    log.info(
        "\n[Question]:\n{}\n[Document]:\n{}\n[RetrievalGrader result]:\n{}",
        question,
        document,
        answer.binaryScore);
  }

  @Test
  public void testGeneration() {
    EmbeddingSearchResult<TextSegment> relevantDocs = helloEmbeddingStore.search(question);
    List<String> docs =
        relevantDocs.matches().stream().map(m -> m.embedded().text()).collect(Collectors.toList());
    GenerationNodeFn qr = GenerationNodeFn.of(getZhipuAiKey());
    String result = qr.apply(question, docs);
    log.info("\n[Question]:\n{}\n[Generation result]:\n{}", question, result);
  }

  @Test
  public void testQuestionRewriter() {
    QuestionRewriterNodeFn questionRewriterNodeFn = QuestionRewriterNodeFn.of(getZhipuAiKey());
    String result = questionRewriterNodeFn.apply(question);
    log.info("\n[Question]:\n{}\n[QuestionRewriter result]:\n{}", question, result);
  }

  @Test
  public void testGraphing() throws Exception {
    AdaptiveRagGraph adaptiveRagGraph =
        new AdaptiveRagGraph(getZhipuAiKey(), getTavilyApiKey(), helloEmbeddingStore);
    StateGraph<AdaptiveRagGraph.State> graph = adaptiveRagGraph.buildGraph();

    GraphRepresentation plantUml = graph.getGraph(PLANTUML, "Adaptive RAG");
    String plantUmlContent = plantUml.getContent();
    Files.writeString(Path.of("plantUm.puml"), plantUmlContent);
    log.info("plantUml:{}", plantUmlContent);

    GraphRepresentation mermaid = graph.getGraph(MERMAID, "Adaptive RAG");
    String mermaidContent = "```mermaid\n" + mermaid.getContent() + "\n```";
    Files.writeString(Path.of("mermaid.md"), mermaidContent);
    log.info("mermaid{}", mermaidContent);
  }
}
