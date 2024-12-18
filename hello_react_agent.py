import datetime

# from PIL import Image as PilImage
from dotenv import load_dotenv
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.bing_search import BingSearchResults
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.utilities import BingSearchAPIWrapper
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.tools import GoogleSerperResults

load_dotenv()

llm = ChatZhipuAI(model="GLM-4-Plus")

####
# https://python.langchain.com/docs/integrations/tools/

# TAVILY_API_KEY
# https://api.tavily.com
tavily_search = TavilySearchResults(
    api_wrapper=TavilySearchAPIWrapper(), max_results=3)

# SERPER_API_KEY
# https://serper.dev/api-key
serper_search = GoogleSerperResults(
    api_wrapper=GoogleSerperAPIWrapper(), max_results=3)

# https://duckduckgo.com/ 梯子
# duck_duck_go_search = DuckDuckGoSearchResults(max_results=3, output_format="json")

# https://github.com/searxng/searxng
# https://github.com/searxng/searxng-docker
# searx_search = SearxSearchResults(max_results=3)

query = "梅西最近效力的俱乐部"

####

# tavily_result = tavily_search.invoke({"query": query})
# print("tavily_result", tavily_result)
# serper_result = serper_search.invoke({"query": query})
# print("serper_result", serper_result)

####
search_tools = [tavily_search, serper_search]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Current time: {time}"),
    MessagesPlaceholder("messages")
]).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

agent_executor = create_react_agent(
    model=llm,
    tools=search_tools,
    state_modifier=prompt
)

execute_result = agent_executor.invoke(
    {"messages": [("user", query)]}
)

print("execute_result", execute_result)
