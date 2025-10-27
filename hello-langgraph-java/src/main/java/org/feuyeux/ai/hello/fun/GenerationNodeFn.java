package org.feuyeux.ai.hello.fun;

import static org.feuyeux.ai.hello.repository.ChatModelBuilder.buildChatModel;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.V;
import java.util.List;
import java.util.function.BiFunction;
import lombok.Value;

@Value(staticConstructor = "of")
public class GenerationNodeFn implements BiFunction<String, List<String>, String> {

  public interface Service {

    @UserMessage(
        "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n"
            + "Question: {{question}} \n"
            + "Context: {{context}} \n"
            + "Answer:")
    String invoke(@V("question") String question, @V("context") List<String> context);
  }

  String apiKey;

  public String apply(String question, List<String> context) {
    ChatModel chatLanguageModel = buildChatModel(apiKey);
    Service service = AiServices.create(Service.class, chatLanguageModel);
    return service.invoke(question, context); // service
  }
}
