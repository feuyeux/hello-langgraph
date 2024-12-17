import asyncio
import io
import operator
import os
from typing import Annotated, List, Tuple
from typing import Union

# from PIL import Image as PilImage
from dotenv import load_dotenv
from langchain import hub
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_openai import ChatOpenAI
from langgraph.graph import END
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

# Define the models
zhipu_model = ChatZhipuAI(model="GLM-4-Plus")
# llama_model = ChatOpenAI(
#     model="llama3.3", base_url="http://localhost:11434/v1")
# kimi_model = ChatOpenAI(
#     model="moonshot-v1-8k",
#     api_key=os.environ["MOONSHOT_API_KEY"],
#     base_url="https://api.moonshot.cn/v1",
# )
llm = zhipu_model

# Define the tools
# https://api.tavily.com
tavily = TavilySearchResults(max_results=3)

# https://duckduckgo.com/
duck_duck_go = DuckDuckGoSearchResults(max_results=3, output_format="json")

search_tools = [tavily]

# Define the executor agent

# https://smith.langchain.com/hub
# https://smith.langchain.com/hub/ih/ih-react-agent-executor
# prompt = hub.pull("ih/ih-react-agent-executor")
# prompt.pretty_print()
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("messages")
])

agent_executor = create_react_agent(
    model=llm,
    tools=search_tools,
    state_modifier=prompt
)


# execute_result = agent_executor.invoke(
#     {"messages": [("user", "What is the league ranking of the club that the World Cup champion captain has played for in the past three years?")]}
# )
# print("execute_result", execute_result)


# Define the plan agent

class Plan(BaseModel):
    """Plan to follow in future"""
    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
        ),
        ("placeholder", "{messages}"),
    ]
)
planner = planner_prompt | llm.with_structured_output(Plan)

# planner_result = planner.invoke({
#     "messages": [
#         ("user",
#          "What is the league ranking of the club that the World Cup champion captain has played for in the past three years?")
#     ],
#     "steps": ["Get World Cup Champion", "Get League Ranking"]
# })
# print("planner_result", planner_result)


# Define the replan agent

class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
                    "If you need to further use tools to get the answer, use Plan."
    )


replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)

replanner = replanner_prompt | llm.with_structured_output(Act)


# Define the workflow


class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


async def execute_step(state: PlanExecute):
    plan = state["plan"]
    plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
    if not plan:
        return {
            "past_steps": [("No task", "Error: Plan is empty")],
        }
    else:
        task = plan[0]
        task_formatted = f"""For the following plan:
    {plan_str}\n\nYou are tasked with executing step {1}, {task}."""
        try:
            msg = ("user", task_formatted)
            # print("msg:", msg)
            agent_response = await agent_executor.ainvoke(
                {"messages": [msg]}
            )
            execute_result = agent_response["messages"][-1].content
            return {
                "past_steps": [(task, execute_result)],
            }
        except Exception as e:
            print(f"Error executing step: {e}")
            return {
                "past_steps": [(task, f"Error: {e}")],
            }


async def plan_step(state: PlanExecute):
    plan = await planner.ainvoke({"messages": [("user", state["input"])]})
    return {"plan": plan.steps}


async def replan_step(state: PlanExecute):
    output = await replanner.ainvoke(state)
    if output is None:
        return {"response": "Error: replanner returned None"}
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}


def should_end(state: PlanExecute):
    if "response" in state and state["response"]:
        return END
    else:
        return "execute_node"


workflow = StateGraph(PlanExecute)

# Add the plan node
workflow.add_node("plan_node", plan_step)

# Add the execution step
workflow.add_node("execute_node", execute_step)

# Add a replan node
workflow.add_node("replan_node", replan_step)

workflow.add_edge(START, "plan_node")

# From plan we go to agent
workflow.add_edge("plan_node", "execute_node")

# From agent, we replan_node
workflow.add_edge("execute_node", "replan_node")

workflow.add_conditional_edges(
    "replan_node",
    # Next, we pass in the function that will determine which node is called next.
    should_end,
    ["execute_node", END],
)

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile()

# image_bytes = app.get_graph(xray=True).draw_mermaid_png()
# image_stream = io.BytesIO(image_bytes)
# image = PilImage.open(image_stream)
# image.show()

# https: // langchain-ai.github.io/langgraph/troubleshooting/errors/GRAPH_RECURSION_LIMIT/
config = {"recursion_limit": 100}
inputs = {"input": "最近一次世界杯的冠军队队长过去一年所效力的俱乐部在联赛中的排名？"}


async def main():
    async for event in app.astream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(k, "::::", v)


asyncio.run(main())
