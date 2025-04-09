package org.feuyeux.ai.hello;

import static org.feuyeux.ai.hello.service.LanggraphService.getZhipuAiKey;
import static org.junit.jupiter.api.Assertions.assertEquals;

import lombok.extern.slf4j.Slf4j;
import org.feuyeux.ai.hello.fun.StructuredOutputFn;
import org.feuyeux.ai.hello.fun.StructuredOutputFn.ActionType;
import org.feuyeux.ai.hello.fun.StructuredOutputFn.BizAction;
import org.feuyeux.ai.hello.fun.StructuredOutputFn.ScenarioType;
import org.feuyeux.ai.hello.util.DotEnvConfig;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

@Slf4j
public class StructuredOutputTests {
  StructuredOutputFn structuredOutputFn =new StructuredOutputFn(getZhipuAiKey());

  @BeforeAll
  public static void beforeAll() throws Exception {
    DotEnvConfig.load();
  }

  // mvn test -Dtest=org.feuyeux.ai.hello.StructedOutputTests.testOutput
  @Test
  public void testOutput() {
    // Define the questions and expected responses mapping
    String[] questions = {
      "Give me a relax song",
      "It's too hot in here",
      "I want to go to the nearest gas station",
      "I can't hear the music",
      "Open the windows"
    };

    // Create the expected BizAction responses
    BizAction[] expectedResponses = {
      createBizAction(ScenarioType.media, ActionType.play),
      createBizAction(ScenarioType.air_conditioning_control, ActionType.decrease),
      createBizAction(ScenarioType.navigation, ActionType.navigate),
      createBizAction(ScenarioType.volume_control, ActionType.increase),
      createBizAction(ScenarioType.vehicle_control, ActionType.open)
    };

    // Process each question
    for (int i = 0; i < questions.length; i++) {
      String question = questions[i];
      try {
        StructuredOutputFn.BizAction actualAction = structuredOutputFn.apply(question);
        BizAction expectedAction = expectedResponses[i];
        assertEquals(expectedAction.scenario, actualAction.scenario);
      } catch (Exception e) {
        log.error("Error processing question '{}': {}", question, e.getMessage());
      }
    }
  }

  // Helper method to create BizAction objects
  private BizAction createBizAction(ScenarioType scenario, ActionType action) {
    BizAction bizAction = new BizAction();
    bizAction.scenario = scenario;
    bizAction.action = action;
    return bizAction;
  }
}
