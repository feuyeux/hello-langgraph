package org.feuyeux.ai.hello.repository;

import dev.langchain4j.model.zhipu.ZhipuAiEmbeddingModel;
import java.time.Duration;

public class EmbeddingModelBuilder {

  public static ZhipuAiEmbeddingModel buildEmbeddingModel(String apiKey) {
    return ZhipuAiEmbeddingModel.builder()
        .apiKey(apiKey)
        .connectTimeout(Duration.ofMinutes(1))
        .callTimeout(Duration.ofMinutes(2))
        .readTimeout(Duration.ofMinutes(2))
        .writeTimeout(Duration.ofMinutes(2))
        .build();
  }
}
