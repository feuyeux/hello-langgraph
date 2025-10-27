import datetime

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

llm = ChatOllama(model="qwen2.5")

####
# 使用 DuckDuckGo 搜索（不需要 API key）
search = DuckDuckGoSearchResults(max_results=3)
search_tools = [search]

query = "梅西最近效力的俱乐部"

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
