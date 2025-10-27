package org.feuyeux.ai.hello.repository;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.ollama.OllamaChatModel;
import java.time.Duration;

public class ChatModelBuilder {

  public static ChatModel buildChatModel(String baseUrl) {
    ChatModel chatLanguageModel =
        OllamaChatModel.builder()
            .baseUrl(baseUrl)
            .modelName("qwen2.5")
            .timeout(Duration.ofMinutes(2))
            .logRequests(true)
            .logResponses(true)
            .temperature(0.0)
            .build();
    return chatLanguageModel;
  }
}
