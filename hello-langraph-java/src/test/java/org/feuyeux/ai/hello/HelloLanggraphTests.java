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
    DotEnvConfig.load();
  }

  @Test
  public void testLog() {
    log.error("error");
  }

  @Test
  public void testGraphRun() {
    String result = null;
    try {
      result = langgraphService.generate(question);
    } catch (Exception e) {
      log.error("", e);
    }
    log.info("[Question]:\n{} \n[Graph result]:{}", question, result);
  }

  // ----

  @Test
  public void testQuestionRouter() {
    QuestionRouter qr = QuestionRouter.of(getZhipuAiKey());
    QuestionRouter.Type result = qr.apply("What are the stock options?");
    assertEquals(QuestionRouter.Type.web_search, result);
    result = qr.apply(question);
    assertEquals(QuestionRouter.Type.vectorstore, result);
  }

  @Test
  public void testWebSearch() {
    WebSearchToolFn webSearchTool = WebSearchToolFn.of(getTavilyApiKey());
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
    RetrievalGrader grader = RetrievalGrader.of(getZhipuAiKey());
    // retrieve
    EmbeddingSearchResult<TextSegment> relevant = helloEmbeddingStore.search(question);
    List<EmbeddingMatch<TextSegment>> matches = relevant.matches();
    assertEquals(1, matches.size());
    String document = matches.getFirst().embedded().text();
    // grade
    RetrievalGrader.Arguments arguments = RetrievalGrader.Arguments.of(question, document);
    RetrievalGrader.Score answer = grader.apply(arguments);
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
    Generation qr = Generation.of(getZhipuAiKey());
    String result = qr.apply(question, docs);
    log.info("\n[Question]:\n{}\n[Generation result]:\n{}", question, result);
  }

  @Test
  public void testQuestionRewriter() {
    QuestionRewriter questionRewriter = QuestionRewriter.of(getZhipuAiKey());
    String result = questionRewriter.apply(question);
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
