package org.feuyeux.ai.hello.fun;

import static org.feuyeux.ai.hello.repository.ChatModelBuilder.buildChatLanguageModel;

import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.output.structured.Description;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.SystemMessage;
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
        "You are an expert at routing a user question to a vectorstore or web search.\n"
            + "The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.\n"
            + "Use the vectorstore for questions on these topics. Otherwise, use web-search.")
    Result invoke(String question);
  }

  String apiKey;

  @Override
  public Type apply(String question) {
    ChatLanguageModel chatLanguageModel = buildChatLanguageModel(apiKey);
    Service extractor = AiServices.create(Service.class, chatLanguageModel);
    Result ds = extractor.invoke(question);
    return ds.datasource;
  }
}