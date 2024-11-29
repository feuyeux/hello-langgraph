from pprint import pprint
from graph_workflow import build_workflow
from IPython.display import Image, display

if __name__ == '__main__':
    graph = build_workflow().compile()
    display(Image(graph.get_graph().draw_mermaid_png()))
    questions = [
        "What are the types of agent memory?",
        "How does the AlphaCodium paper work?"
    ]
    for question in questions:
        inputs = {"question": question}
        for output in graph.stream(inputs):
            for key, value in output.items():
                # Node
                pprint(f"Node '{key}':")
                # Optional: print full state at each node
                # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
            pprint("\n---\n")
        # Final generation
        pprint(value["generation"])
