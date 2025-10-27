package org.feuyeux.ai.hello.service;

import static org.bsc.langgraph4j.utils.CollectionsUtils.mapOf;

import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.bsc.langgraph4j.CompiledGraph;
import org.feuyeux.ai.hello.graph.AdaptiveRagGraph;
import org.feuyeux.ai.hello.repository.HelloEmbeddingStore;
import org.feuyeux.ai.hello.util.DotEnvConfig;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
@Slf4j
public class LanggraphService {
  @Autowired private HelloEmbeddingStore helloEmbeddingStore;

  private CompiledGraph<AdaptiveRagGraph.State> graph;

  // Ollama base URL
  public static String getOllamaBaseUrl() {
    return DotEnvConfig.valueOf("OLLAMA_BASE_URL").orElse("http://localhost:11434");
  }

  // https://app.tavily.com/
  public static String getTavilyApiKey() {
    return DotEnvConfig.valueOf("TAVILY_API_KEY")
        .orElseThrow(() -> new IllegalArgumentException("no TAVILY APIKEY provided!"));
  }

  public String generate(String question) throws Exception {
    org.bsc.async.AsyncGenerator<org.bsc.langgraph4j.NodeOutput<AdaptiveRagGraph.State>> result =
        graph.stream(mapOf("question", question));
    StringBuilder generationBuilder = new StringBuilder();
    for (org.bsc.langgraph4j.NodeOutput<AdaptiveRagGraph.State> r : result) {
      log.info("Node: '{}':\n", r.node());
      generationBuilder.append(r.state().generation().orElse(""));
    }
    return generationBuilder.toString();
  }

  @PostConstruct
  public void init() {
    try {
      AdaptiveRagGraph adaptiveRagGraph =
          new AdaptiveRagGraph(getOllamaBaseUrl(), getTavilyApiKey(), helloEmbeddingStore);
      graph = adaptiveRagGraph.buildGraph().compile();
    } catch (Exception e) {
      log.error("", e);
    }
  }
}
