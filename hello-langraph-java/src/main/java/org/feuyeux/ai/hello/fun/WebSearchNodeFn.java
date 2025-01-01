package org.feuyeux.ai.hello.fun;

import dev.langchain4j.rag.content.Content;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;
import java.util.List;
import java.util.function.Function;
import lombok.Value;

@Value(staticConstructor = "of")
public class WebSearchNodeFn implements Function<String, List<Content>> {
  String tavilyApiKey;

  @Override
  public List<Content> apply(String query) {
    WebSearchEngine webSearchEngine = TavilyWebSearchEngine.builder().apiKey(tavilyApiKey).build();
    ContentRetriever webSearchContentRetriever =
        WebSearchContentRetriever.builder().webSearchEngine(webSearchEngine).maxResults(3).build();
    return webSearchContentRetriever.retrieve(new Query(query));
  }
}
