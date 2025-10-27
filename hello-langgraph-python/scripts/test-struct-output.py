from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


llm = ChatOllama(model="qwen2.5")


class BizAction(BaseModel):
    scenario: Literal["navigation", "media", "air conditioning control", "volume control", "vehicle control"] = Field(
        ...,
        description="Given a user question choose to a special scenario.",
    )
    action: Literal["turn on", "turn off", "increase", "decrease", "play", "stop", "pause", "next", "open", "close", "navigate"] = Field(
        ...,
        description="The action to be taken.",
    )


action_schema = llm.with_structured_output(BizAction)

system = """You are a smart carbin assistant that maps user's {question} to specific scenarios and actions.
For questions about:
- Music or songs -> map to "media" scenario
- Navigation or directions -> map to "navigation" scenario
- Temperature or AC -> map to "air conditioning control" scenario
- Sound or volume -> map to "volume control" scenario
- Windows, doors, or other car controls -> map to "vehicle control" scenario

Choose the most appropriate action from: "turn on", "turn off", "increase", "decrease", "set"."""

biz_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{question}")
])

biz_chain = biz_prompt | action_schema

try:
    questions = [
        "Give me a relax song",
        "It's too hot in here",
        "I want to go to the nearest gas station",
        "I can't hear the music",
        "Open the windows"
    ]
    for i, question in enumerate(questions, 1):
        biz_action = biz_chain.invoke({"question": question})
        print(f"{i}. Question:{question}, Action: {biz_action}")

except Exception as e:
    print(f"Error processing request: {str(e)}")
