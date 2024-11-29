from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import OllamaLLM
from langgraph.graph import MessagesState

from tools import *

prompt = "You are a helpful assistant tasked with using performing arithmetic on a set of inputs."
sys_msg = SystemMessage(content=prompt)

tools = [add, multiply, divide]
llm = OllamaLLM(model="llama3.2", temperature=0)


def invoke_tool(tool_name, *args):
    for tool in tools:
        if tool.__name__ == tool_name:
            return tool(*args)
    raise ValueError(f"Tool {tool_name} not found")


def reasoner(state: MessagesState):
    messages = [sys_msg] + state["messages"]
    response = llm.invoke(messages)
    # Check if the response is a tool call
    if isinstance(response, dict) and "tool" in response:
        tool_name = response["tool"]
        tool_args = response.get("args", [])
        tool_result = invoke_tool(tool_name, *tool_args)
        return {"messages": [HumanMessage(content=str(tool_result))]}
    return {"messages": [response]}
