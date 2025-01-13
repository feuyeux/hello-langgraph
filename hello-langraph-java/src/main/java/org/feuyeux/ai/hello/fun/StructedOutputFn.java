package org.feuyeux.ai.hello.fun;

import static org.feuyeux.ai.hello.repository.ChatModelBuilder.buildChatLanguageModel;

import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.output.structured.Description;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.SystemMessage;
import java.util.function.Function;
import lombok.ToString;
import lombok.Value;

@Value(staticConstructor = "of")
public class StructedOutputFn implements Function<String, StructedOutputFn.BizAction> {
  String apiKey;

  @Override
  public StructedOutputFn.BizAction apply(String question) {
    ChatLanguageModel chatLanguageModel = buildChatLanguageModel(apiKey);
    Service service = AiServices.create(Service.class, chatLanguageModel);
    return service.invoke(question);
  }

  interface Service {
    @SystemMessage(
        """
                              You are a smart carbin assistant that maps user's {question} to specific scenarios and actions.
                              For questions about:
                              - Music or songs -> map to "media" scenario
                              - Navigation or directions -> map to "navigation" scenario
                              - Temperature or AC -> map to "air conditioning control" scenario
                              - Sound or volume -> map to "volume control" scenario
                              - Windows, doors, or other car controls -> map to "vehicle control" scenario

                              Choose the most appropriate action from: "turn on", "turn off", "increase", "decrease", "set".
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
