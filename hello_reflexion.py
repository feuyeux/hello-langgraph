from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langchain_core.tools import StructuredTool
import json
import datetime
from pydantic import BaseModel, Field
from pydantic import ValidationError
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.messages import ToolMessage
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

load_dotenv()

MAX_CRITIQUES = 1
MAX_ITERATIONS = 2
model_name = "llama3.3"
llm = ChatOpenAI(
    model=model_name,
    base_url="http://localhost:11434/v1",
)

# https://api.tavily.com
tavily = TavilySearchResults(
    api_wrapper=TavilySearchAPIWrapper(), max_results=5)

# https://duckduckgo.com/
duck_duck_go = DuckDuckGoSearchResults(
    api_wrapper=DuckDuckGoSearchAPIWrapper(), max_results=5)

search_tools = [duck_duck_go, tavily]


class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous")


class AnswerQuestion(BaseModel):
    """Answer the question. Provide an answer, reflection, and then follow up with search queries to improve the answer."""

    answer: str = Field(
        description="~250 word detailed answer to the question.")
    reflection: Reflection = Field(
        description="Your reflection on the initial answer.")
    search_queries: list[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )


class ResponderWithRetries:
    def __init__(self, runnable, validator):
        self.runnable = runnable
        self.validator = validator

    def respond(self, state: list):
        response = []
        for attempt in range(MAX_CRITIQUES):
            start_time = datetime.datetime.now()
            response = self.runnable.invoke(
                {"messages": state["messages"]},
                {"tags": [f"attempt:{attempt}"]}
            )
            end_time = datetime.datetime.now()
            print(f"{attempt} RESPONSE({end_time-start_time} ms):")
            print(response)
            try:
                self.validator.invoke(response)
                print("VALIDATED")
                return {"messages": response}
            except ValidationError as e:
                # schema_json = self.validator.schema_json()
                schema_json = json.dumps(self.validator.model_json_schema())
                message = ToolMessage(
                    content=f"{
                        repr(e)}\n\nPay close attention to the function schema.\n\n"
                    + schema_json
                    + " Respond by fixing all validation errors.",
                    tool_call_id=response.tool_calls[0]["id"],
                )
                state["messages"] += [response, message]
                print(f"{attempt} STATE:")
                print(state)
        return {"messages": response}


actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "user",
            "\n\n<system>Reflect on the user's original question and the"
            " actions taken thus far. Respond using the {function_name} function.</reminder>",
        ),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

initial_answer_chain = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer.",
    function_name=AnswerQuestion.__name__,
) | llm.bind_tools(tools=[AnswerQuestion])

validator = PydanticToolsParser(tools=[AnswerQuestion])

first_responder = ResponderWithRetries(
    runnable=initial_answer_chain, validator=validator
)

# 修订指令
revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""


# 修订答案
# Citations 引用
# Extend the initial answer schema to include references.
# Forcing citation in the model encourages grounded responses
class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question. Provide an answer, reflection,

    cite your reflection with references, and finally
    add search queries to improve the answer."""

    references: list[str] = Field(
        description="Citations motivating your updated answer."
    )


revision_chain = actor_prompt_template.partial(
    first_instruction=revise_instructions,
    function_name=ReviseAnswer.__name__,
) | llm.bind_tools(tools=[ReviseAnswer])
revision_validator = PydanticToolsParser(tools=[ReviseAnswer])

revisor = ResponderWithRetries(
    runnable=revision_chain, validator=revision_validator)


def run_queries(search_queries: list[str], **kwargs):
    """Run the generated queries."""
    return search_tools[0].batch([{"query": query} for query in search_queries])


def run_queries2(search_queries: list[str], **kwargs):
    """Run the generated queries."""
    return search_tools[1].batch([{"query": query} for query in search_queries])

tool_node = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries2, name=ReviseAnswer.__name__),
    ]
)


class State(TypedDict):
    messages: Annotated[list, add_messages]


builder = StateGraph(State)
builder.add_node("draft", first_responder.respond)

builder.add_node("execute_tools", tool_node)
builder.add_node("revise", revisor.respond)
# draft -> execute_tools
builder.add_edge("draft", "execute_tools")
# execute_tools -> revise
builder.add_edge("execute_tools", "revise")


# Define looping logic: if we have revised N times, stop
def _get_num_iterations(state: list):
    i = 0
    for m in state[::-1]:
        if m.type not in {"tool", "ai"}:
            break
        i += 1
    return i


def event_loop(state: list):
    # in our case, we'll just stop after N plans
    num_iterations = _get_num_iterations(state["messages"])
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"


# revise -> execute_tools OR end
builder.add_conditional_edges("revise", event_loop, ["execute_tools", END])
builder.add_edge(START, "draft")
graph = builder.compile()
query = "How to improve the employment environment next year?"
events = graph.stream(
    {"messages": [("user", query)]},
    stream_mode="values",
)
for i, step in enumerate(events):
    print(f"Step {i}")
    step["messages"][-1].pretty_print()
