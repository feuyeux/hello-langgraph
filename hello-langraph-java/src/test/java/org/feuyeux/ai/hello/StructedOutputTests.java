package org.feuyeux.ai.hello;

import static org.feuyeux.ai.hello.service.LanggraphService.getZhipuAiKey;

import lombok.extern.slf4j.Slf4j;
import org.feuyeux.ai.hello.fun.StructedOutputFn;
import org.feuyeux.ai.hello.util.DotEnvConfig;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

@Slf4j
public class StructedOutputTests {
  @BeforeAll
  public static void beforeAll() throws Exception {
    DotEnvConfig.load();
  }

  @Test
  public void testOutput() {
    StructedOutputFn structedOutputFn = StructedOutputFn.of(getZhipuAiKey());
    String[] questions =
        new String[] {
          "Give me a relax song",
          "It's too hot in here",
          "I want to go to the nearest gas station",
          "I can't hear the music",
          "Open the windows"
        };
    for (int i = 0; i < questions.length; i++) {
      String question = questions[i];
      StructedOutputFn.BizAction bizAction = structedOutputFn.apply(question);
      log.info("{}. Question:{},Action:{}", i, question, bizAction);
    }
  }
}
