package org.feuyeux.ai.hello.fun;

import static org.bsc.langgraph4j.utils.CollectionsUtils.mapOf;
import static org.feuyeux.ai.hello.repository.ChatModelBuilder.buildChatLanguageModel;

import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.SystemMessage;
import java.util.function.Function;
import lombok.Value;

@Value(staticConstructor = "of")
public class QuestionRewriter implements Function<String, String> {

  String apiKey;

  interface LLMService {
    @SystemMessage(
        "You a question re-writer that converts an input question to a better version that is optimized \n"
            + "for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning.")
    String invoke(String question);
  }

  @Override
  public String apply(String question) {
    ChatLanguageModel chatLanguageModel = buildChatLanguageModel(apiKey);
    LLMService service = AiServices.create(LLMService.class, chatLanguageModel);
    PromptTemplate template =
        PromptTemplate.from(
            "Here is the initial question: \n\n {{question}} \n Formulate an improved question.");
    Prompt prompt = template.apply(mapOf("question", question));
    return service.invoke(prompt.text());
  }
}
