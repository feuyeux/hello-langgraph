package org.feuyeux.ai.hello.service;

import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.bsc.langgraph4j.CompiledGraph;
import org.feuyeux.ai.hello.fun.AdaptiveRag;
import org.feuyeux.ai.hello.repository.HelloEmbeddingStore;
import org.feuyeux.ai.hello.util.DotEnvConfig;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
@Slf4j
public class LanggraphService {
  @Autowired private HelloEmbeddingStore helloEmbeddingStore;

  private CompiledGraph<AdaptiveRag.State> graph;

  public static String getZhipuAiKey() {
    return DotEnvConfig.valueOf("ZHIPUAI_API_KEY")
        .orElseThrow(() -> new IllegalArgumentException("no ZHIPUAI APIKEY provided!"));
  }

  public static String getTavilyApiKey() {
    return DotEnvConfig.valueOf("TAVILY_API_KEY")
        .orElseThrow(() -> new IllegalArgumentException("no TAVILY APIKEY provided!"));
  }

  @PostConstruct
  public void init() {
    try {
      AdaptiveRag adaptiveRag =
          new AdaptiveRag(getZhipuAiKey(), getTavilyApiKey(), helloEmbeddingStore);
      graph = adaptiveRag.buildGraph().compile();
    } catch (Exception e) {
      log.error("", e);
    }
  }
}
