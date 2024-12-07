from IPython.display import Image
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

from reasoning import reasoner, tools
import os

# Graph
builder = StateGraph(MessagesState)

# Add nodes
builder.add_node("reasoner", reasoner)
builder.add_node("tools", ToolNode(tools))

# Add edges
builder.add_edge(START, "reasoner")
builder.add_conditional_edges(
    "reasoner",
    tools_condition,
)
builder.add_edge("tools", "reasoner")
react_graph = builder.compile()

img = Image(react_graph.get_graph(xray=True).draw_mermaid_png())
with open("re_act_graph.png", "wb") as f:
    f.write(img.data)

messages = [HumanMessage(content="5 * (2 + 3) / 2 = ?")]
messages = react_graph.invoke({"messages": messages})
# Displaying the response
for m in messages['messages']:
    m.pretty_print()
