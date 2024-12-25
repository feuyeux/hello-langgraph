package org.feuyeux.ai.hello.repository;

import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.zhipu.ZhipuAiChatModel;
import java.time.Duration;

public class ChatModelBuilder {

  public static ChatLanguageModel buildChatLanguageModel(String apiKey) {
    ChatLanguageModel chatLanguageModel =
        ZhipuAiChatModel.builder()
            .apiKey(apiKey)
            .model("GLM-4-Plus")
            .connectTimeout(Duration.ofMinutes(1))
            .callTimeout(Duration.ofMinutes(2))
            .readTimeout(Duration.ofMinutes(2))
            .writeTimeout(Duration.ofMinutes(2))
            .logRequests(true)
            .logResponses(true)
            .maxRetries(2)
            .temperature(0.0)
            .maxToken(1000)
            .build();
    return chatLanguageModel;
  }
}
