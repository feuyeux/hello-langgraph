package org.feuyeux.ai.hello.fun;

import static org.feuyeux.ai.hello.repository.ChatModelBuilder.buildChatModel;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.output.structured.Description;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;
import java.util.function.Function;
import lombok.Value;
import lombok.extern.slf4j.Slf4j;
import org.feuyeux.ai.hello.graph.AdaptiveRagGraph;

@Slf4j
/** Router for user queries to the most relevant datasource. */
@Value(staticConstructor = "of")
public class QuestionRouterEdgeFn implements Function<String, QuestionRouterEdgeFn.Type> {

  private String routeQuestion(AdaptiveRagGraph.State state) {
    log.debug("---ROUTE QUESTION---");
    String question = state.question();
    QuestionRouterEdgeFn.Type source = QuestionRouterEdgeFn.of(apiKey).apply(question);
    if (source == QuestionRouterEdgeFn.Type.web_search) {
      log.debug("---ROUTE QUESTION TO WEB SEARCH---");
    } else {
      log.debug("---ROUTE QUESTION TO RAG---");
    }
    return source.name();
  }

  public enum Type {
    vectorstore,
    web_search
  }

  /** Route a user query to the most relevant datasource. */
  static class Result {
    @Description("Given a user question choose to route it to web search or a vectorstore.")
    Type datasource;
  }

  interface Service {
    @SystemMessage(
        """
        You are an expert at routing a user question to a vectorstore or web search.
        The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.

        ROUTING RULES:
        - If the question is about AI agents, agent memory, agent planning, agent tools, or agent systems -> use "vectorstore"
        - If the question is about prompt engineering, prompting techniques, or prompt design -> use "vectorstore"
        - If the question is about adversarial attacks on LLMs or AI security -> use "vectorstore"
        - For all other questions (current events, general knowledge, specific facts) -> use "web_search"

        Your task is to analyze the question and return a JSON object with the following structure:
        {
          "datasource": "vectorstore|web_search"
        }

        IMPORTANT: Return ONLY a valid JSON object, without any additional text or explanations.
        """)
    @UserMessage(
        "Please analyze this question and determine the appropriate datasource: {question}")
    Result invoke(String question);
  }

  String apiKey;

  @Override
  public Type apply(String question) {
    ChatModel chatLanguageModel = buildChatModel(apiKey);
    Service extractor = AiServices.create(Service.class, chatLanguageModel);
    try {
      Result ds = extractor.invoke(question);
      return ds.datasource;
    } catch (Exception e) {
      log.error("Error routing question: {}", e.getMessage());
      // Default to vectorstore when JSON parsing fails
      return Type.vectorstore;
    }
  }
}
