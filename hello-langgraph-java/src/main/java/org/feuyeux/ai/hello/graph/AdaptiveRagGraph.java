package org.feuyeux.ai.hello.graph;

import static java.util.Collections.emptyList;
import static org.bsc.langgraph4j.StateGraph.END;
import static org.bsc.langgraph4j.StateGraph.START;
import static org.bsc.langgraph4j.action.AsyncEdgeAction.edge_async;
import static org.bsc.langgraph4j.action.AsyncNodeAction.node_async;
import static org.bsc.langgraph4j.utils.CollectionsUtils.listOf;
import static org.bsc.langgraph4j.utils.CollectionsUtils.mapOf;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.stream.Collectors;
import lombok.extern.slf4j.Slf4j;
import org.bsc.langgraph4j.StateGraph;
import org.bsc.langgraph4j.state.AgentState;
import org.feuyeux.ai.hello.fun.*;
import org.feuyeux.ai.hello.repository.HelloEmbeddingStore;

@Slf4j(topic = "AdaptiveRag")
public class AdaptiveRagGraph {

  private final String aiApiKey;
  private final String tavilyApiKey;
  private HelloEmbeddingStore helloEmbeddingStore;

  public AdaptiveRagGraph(
      String aiApiKey, String tavilyApiKey, HelloEmbeddingStore helloEmbeddingStore) {
    this.aiApiKey = aiApiKey;
    this.tavilyApiKey = tavilyApiKey;
    this.helloEmbeddingStore = helloEmbeddingStore;
  }

  public StateGraph<State> buildGraph() throws Exception {
    return new StateGraph<>(State::new)
        .addConditionalEdges(
            START,
            edge_async(this::routeQuestion),
            mapOf(
                "web_search", "web_search",
                "vectorstore", "retrieve"))
        .addNode("web_search", node_async(this::webSearch))
        .addNode("retrieve", node_async(this::retrieve))
        .addEdge("web_search", "generate")
        .addNode("generate", node_async(this::generate))
        .addEdge("retrieve", "grade_documents")
        .addNode("grade_documents", node_async(this::gradeDocuments))
        .addConditionalEdges(
            "generate",
            edge_async(this::gradeGeneration_v_documentsAndQuestion),
            mapOf(
                "not supported", "generate",
                "useful", END,
                "not useful", "transform_query"))
        .addNode("transform_query", node_async(this::transformQuery))
        .addConditionalEdges(
            "grade_documents",
            edge_async(this::decideToGenerate),
            mapOf(
                "transform_query", "transform_query",
                "generate", "generate"))
        .addEdge("transform_query", "retrieve");
  }

  /**
   * Edge: Route question to web search or RAG.
   *
   * @param state The current graph state
   * @return Next node to call
   */
  private String routeQuestion(AdaptiveRagGraph.State state) {
    log.debug("---ROUTE QUESTION---");
    String question = state.question();
    QuestionRouterEdgeFn.Type source = QuestionRouterEdgeFn.of(aiApiKey).apply(question);
    if (source == QuestionRouterEdgeFn.Type.web_search) {
      log.debug("---ROUTE QUESTION TO WEB SEARCH---");
    } else {
      log.debug("---ROUTE QUESTION TO RAG---");
    }
    return source.name();
  }

  /**
   * Node: Web search based on the re-phrased question.
   *
   * @param state The current graph state
   * @return Updates documents key with appended web results
   */
  private Map<String, Object> webSearch(State state) {
    log.debug("---WEB SEARCH---");
    String question = state.question();
    List<Content> result = WebSearchNodeFn.of(tavilyApiKey).apply(question);
    String webResult =
        result.stream()
            .map(content -> content.textSegment().text())
            .collect(Collectors.joining("\n"));
    return mapOf("documents", listOf(webResult));
  }

  /**
   * Node: Retrieve documents
   *
   * @param state The current graph state
   * @return New key added to state, documents, that contains retrieved documents
   */
  private Map<String, Object> retrieve(State state) {
    log.debug("---RETRIEVE---");
    String question = state.question();
    EmbeddingSearchResult<TextSegment> relevant = helloEmbeddingStore.search(question);
    List<String> documents =
        relevant.matches().stream().map(m -> m.embedded().text()).collect(Collectors.toList());
    return mapOf("documents", documents, "question", question);
  }

  /**
   * Node: Determines whether the retrieved documents are relevant to the question.
   *
   * @param state The current graph state
   * @return Updates documents key with only filtered relevant documents
   */
  private Map<String, Object> gradeDocuments(State state) {
    log.debug("---CHECK DOCUMENT RELEVANCE TO QUESTION---");
    String question = state.question();
    List<String> documents = state.documents();
    final RetrievalGraderNodeFn grader = RetrievalGraderNodeFn.of(aiApiKey);
    List<String> filteredDocs =
        documents.stream()
            .filter(
                d -> {
                  RetrievalGraderNodeFn.Arguments arguments =
                      RetrievalGraderNodeFn.Arguments.of(question, d);
                  RetrievalGraderNodeFn.Score score = grader.apply(arguments);
                  boolean relevant = score.binaryScore.equals("yes");
                  if (relevant) {
                    log.debug("---GRADE: DOCUMENT RELEVANT---");
                  } else {
                    log.debug("---GRADE: DOCUMENT NOT RELEVANT---");
                  }
                  return relevant;
                })
            .collect(Collectors.toList());
    return mapOf("documents", filteredDocs);
  }

  /**
   * Node: Generate answer
   *
   * @param state The current graph state
   * @return New key added to state, generation, that contains LLM generation
   */
  private Map<String, Object> generate(State state) {
    log.debug("---GENERATE---");
    String question = state.question();
    List<String> documents = state.documents();
    String generation = GenerationNodeFn.of(aiApiKey).apply(question, documents); // service
    return mapOf("generation", generation);
  }

  /**
   * Node: Transform the query to produce a better question.
   *
   * @param state The current graph state
   * @return Updates question key with a re-phrased question
   */
  private Map<String, Object> transformQuery(State state) {
    log.debug("---TRANSFORM QUERY---");
    String question = state.question();
    String betterQuestion = QuestionRewriterNodeFn.of(aiApiKey).apply(question);
    return mapOf("question", betterQuestion);
  }

  /**
   * Edge: Determines whether to generate an answer, or re-generate a question.
   *
   * @param state The current graph state
   * @return Binary decision for next node to call
   */
  private String decideToGenerate(State state) {
    log.debug("---ASSESS GRADED DOCUMENTS---");
    List<String> documents = state.documents();
    if (documents.isEmpty()) {
      log.debug("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---");
      return "transform_query";
    }
    log.debug("---DECISION: GENERATE---");
    return "generate";
  }

  /**
   * Edge: Determines whether the generation is grounded in the document and answers question.
   *
   * @param state The current graph state
   * @return Decision for next node to call
   */
  private String gradeGeneration_v_documentsAndQuestion(State state) {
    log.debug("---CHECK HALLUCINATIONS---");

    String question = state.question();
    List<String> documents = state.documents();
    String generation =
        state.generation().orElseThrow(() -> new IllegalStateException("generation is not set!"));

    HallucinationGraderEdgeFn.Score score =
        HallucinationGraderEdgeFn.of(aiApiKey)
            .apply(HallucinationGraderEdgeFn.Arguments.of(documents, generation));

    if (Objects.equals(score.binaryScore, "yes")) {
      log.debug("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---");
      log.debug("---GRADE GENERATION vs QUESTION---");
      AnswerGraderEdgeNodeFn.Score score2 =
          AnswerGraderEdgeNodeFn.of(aiApiKey)
              .apply(AnswerGraderEdgeNodeFn.Arguments.of(question, generation));
      if (Objects.equals(score2.binaryScore, "yes")) {
        log.debug("---DECISION: GENERATION ADDRESSES QUESTION---");
        return "useful";
      }

      log.debug("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---");
      return "not useful";
    }

    log.debug("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---");
    return "not supported";
  }

  /**
   * Represents the state of our graph. Attributes: question: question generation: LLM generation
   * documents: list of documents
   */
  public static class State extends AgentState {
    public State(Map<String, Object> initData) {
      super(initData);
    }

    public String question() {
      Optional<String> result = value("question");
      return result.orElseThrow(() -> new IllegalStateException("question is not set!"));
    }

    public Optional<String> generation() {
      return value("generation");
    }

    public List<String> documents() {
      return this.<List<String>>value("documents").orElse(emptyList());
    }
  }
}
