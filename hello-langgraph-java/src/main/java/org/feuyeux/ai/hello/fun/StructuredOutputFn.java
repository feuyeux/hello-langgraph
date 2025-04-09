package org.feuyeux.ai.hello.fun;

import static org.feuyeux.ai.hello.repository.ChatModelBuilder.buildChatLanguageModel;

import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.output.structured.Description;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.SystemMessage;
import java.util.function.Function;

import dev.langchain4j.service.UserMessage;
import jakarta.annotation.PostConstruct;
import lombok.*;
import lombok.extern.slf4j.Slf4j;

@Slf4j

public class StructuredOutputFn implements Function<String, StructuredOutputFn.BizAction> {
  String apiKey;
  ChatLanguageModel chatLanguageModel;

  public StructuredOutputFn(String apiKey) {
    this.apiKey = apiKey;
    chatLanguageModel = buildChatLanguageModel(apiKey);
  }

  @Override
  public StructuredOutputFn.BizAction apply(String question) {
    Service service = AiServices.create(Service.class, chatLanguageModel);
    BizAction action = service.invoke(question);
    log.info("question: {}, action: {}", question, action);
    return action;
  }

  interface Service {
    @SystemMessage(
            """
            You are a smart carbin assistant that maps user's {question} to specific scenarios and actions.
            Your task is to analyze the question and return a JSON object with the following structure:
            {
              "scenario": "media|navigation|air_conditioning_control|volume_control|vehicle_control",
              "action": "turn_on|turn_off|increase|decrease|play|stop|pause|next|open|close|navigate"
            }
    
            For questions about:
            - Music or song to play -> map to "media" scenario
            - Navigation or direction -> map to "navigation" scenario
            - Temperature or AC -> map to "air_conditioning_control" scenario
            - Sound or volume -> map to "volume_control" scenario
            - Windows, doors, or other car controls -> map to "vehicle_control" scenario
    
            Choose the most appropriate action from the listed options.
            IMPORTANT: Return ONLY a valid JSON object, without any additional text or explanations.
            """)
    BizAction invoke(String question);
  }

  @ToString(of = {"scenario", "action"})
  public static class BizAction {
    @Description("Given a user question choose to a special scenario.")
    public ScenarioType scenario;

    @Description("The action to be taken.")
    public ActionType action;
  }

  public enum ScenarioType {
    media,
    navigation,
    air_conditioning_control,
    volume_control,
    vehicle_control
  }

  public enum ActionType {
    turn_on,
    turn_off,
    increase,
    decrease,
    play,
    stop,
    pause,
    next,
    open,
    close,
    navigate
  }
}
