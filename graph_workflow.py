from typing import List

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from graph_nodes import *


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]


def build_workflow():
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("retrieve", retrieve_node)  # retrieve
    workflow.add_node("grade_documents", grade_doc_node)  # grade documents
    workflow.add_node("generate", generate_node)  # generatae
    workflow.add_node("transform_query", transform_query_node)  # transform_query
    workflow.add_node("web_search_node", web_search_node)  # web search

    # Build graph
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate_edge,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "web_search_node")
    workflow.add_edge("web_search_node", "generate")
    workflow.add_edge("generate", END)
    return workflow
