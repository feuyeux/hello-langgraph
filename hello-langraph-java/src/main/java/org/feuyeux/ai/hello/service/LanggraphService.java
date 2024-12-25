package org.feuyeux.ai.hello.service;

import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.bsc.langgraph4j.CompiledGraph;
import org.feuyeux.ai.hello.fun.AdaptiveRag;
import org.feuyeux.ai.hello.repository.HelloEmbeddingStore;
import org.feuyeux.ai.hello.util.DotEnvConfig;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import static org.bsc.langgraph4j.utils.CollectionsUtils.mapOf;

@Service
@Slf4j
public class LanggraphService {
    @Autowired
    private HelloEmbeddingStore helloEmbeddingStore;

    private CompiledGraph<AdaptiveRag.State> graph;

    public static String getZhipuAiKey() {
        return DotEnvConfig.valueOf("ZHIPUAI_API_KEY")
                .orElseThrow(() -> new IllegalArgumentException("no ZHIPUAI APIKEY provided!"));
    }

    public static String getTavilyApiKey() {
        return DotEnvConfig.valueOf("TAVILY_API_KEY")
                .orElseThrow(() -> new IllegalArgumentException("no TAVILY APIKEY provided!"));
    }

    public String generate(String question) throws Exception {
        org.bsc.async.AsyncGenerator<org.bsc.langgraph4j.NodeOutput<AdaptiveRag.State>> result =
                graph.stream(mapOf("question", question));
        StringBuilder generationBuilder = new StringBuilder();
        for (org.bsc.langgraph4j.NodeOutput<AdaptiveRag.State> r : result) {
            log.info("Node: '{}':\n", r.node());
            generationBuilder.append(r.state().generation().orElse(""));
        }
        return generationBuilder.toString();
    }

    @PostConstruct
    public void init() {
        try {
            AdaptiveRag adaptiveRag =                    new AdaptiveRag(getZhipuAiKey(), getTavilyApiKey(), helloEmbeddingStore);
            graph = adaptiveRag.buildGraph().compile();
        } catch (Exception e) {
            log.error("", e);
        }
    }
}
