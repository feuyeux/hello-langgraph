package org.feuyeux.ai.hello.repository;

import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.ollama.OllamaEmbeddingModel;
import java.time.Duration;

public class EmbeddingModelBuilder {

  static EmbeddingModel buildEmbeddingModel(String baseUrl) {
    return OllamaEmbeddingModel.builder()
        .baseUrl(baseUrl)
        .modelName("qwen2.5")
        .timeout(Duration.ofMinutes(2))
        .build();
  }
}
