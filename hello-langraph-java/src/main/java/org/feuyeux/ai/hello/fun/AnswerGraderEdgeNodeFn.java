package org.feuyeux.ai.hello.fun;

import static org.feuyeux.ai.hello.repository.ChatModelBuilder.buildChatLanguageModel;

import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.structured.StructuredPrompt;
import dev.langchain4j.model.input.structured.StructuredPromptProcessor;
import dev.langchain4j.model.output.structured.Description;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.SystemMessage;
import java.util.function.Function;
import lombok.Value;

@Value(staticConstructor = "of")
public class AnswerGraderEdgeNodeFn
    implements Function<AnswerGraderEdgeNodeFn.Arguments, AnswerGraderEdgeNodeFn.Score> {
  /** Binary score to assess answer addresses question. */
  public static class Score {
    @Description("Answer addresses the question, 'yes' or 'no'")
    public String binaryScore;
  }

  @StructuredPrompt("User question: \n\n {{question}} \n\n LLM generation: {{generation}}")
  @Value(staticConstructor = "of")
  public static class Arguments {
    String question;
    String generation;
  }

  interface Service {
    @SystemMessage(
        "You are a grader assessing whether an answer addresses and/or resolves a question. \n\n"
            + "Give a binary score 'yes' or 'no'. Yes, means that the answer resolves the question otherwise return 'no'")
    Score invoke(String userMessage);
  }

  String apiKey;

  @Override
  public Score apply(Arguments args) {
    ChatLanguageModel chatLanguageModel = buildChatLanguageModel(apiKey);
    Service service = AiServices.create(Service.class, chatLanguageModel);
    Prompt prompt = StructuredPromptProcessor.toPrompt(args);
    return service.invoke(prompt.text());
  }
}
