import os

from langchain import hub
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

vector_store = None
os.environ['USER_AGENT'] = 'MyApp/1.0'


def retrieve_node(state):
    print("---RETRIEVE---")
    question = state["question"]

    global vector_store
    if vector_store is None:
        url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
        loader = WebBaseLoader(url)
        docs = loader.load()
        # Split
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500, chunk_overlap=100
        )
        all_splits = text_splitter.split_documents(docs)

        model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
        gpt4all_kwargs = {'allow_download': 'True'}
        embedding = GPT4AllEmbeddings(
            model_name=model_name,
            gpt4all_kwargs=gpt4all_kwargs
        )

        vector_store = Chroma.from_documents(
            documents=all_splits,
            collection_name="rag-chroma",
            embedding=embedding,
        )
    retriever = vector_store.as_retriever()
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def grade_doc_node(state):
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n
        If the document contains keywords related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
        input_variables=["question", "document"],
    )
    llm = OllamaLLM(model="llama3.2", temperature=0)
    retrieval_grader = prompt | llm | JsonOutputParser()

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


def generate_node(state):
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    prompt = hub.pull("rlm/rag-prompt")
    llm = OllamaLLM(model="llama3.2", format="json", temperature=0)
    rag_chain = prompt | llm | StrOutputParser()
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def transform_query_node(state):
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    re_write_prompt = PromptTemplate(
        template="""You a question re-writer that converts an input question to a better version that is optimized \n 
         for vectorstore retrieval. Look at the initial and formulate an improved question. \n
         Here is the initial question: \n\n {question}. Improved question with no preamble: \n """,
        input_variables=["generation", "question"],
    )
    llm = OllamaLLM(model="llama3.2", temperature=0)
    question_rewriter = re_write_prompt | llm | StrOutputParser()

    question_rewriter.invoke({"question": question})
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


def web_search_node(state):
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Web search https://app.tavily.com/
    os.environ['TAVILY_API_KEY'] = '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'
    web_search_tool = TavilySearchResults(k=3)
    docs = web_search_tool.invoke({"query": question})

    # Check if docs is a list and contains dictionaries
    if isinstance(docs, list) and all(isinstance(d, dict) for d in docs):
        web_results = "\n".join([d["content"] for d in docs])
    else:
        print("Error: Expected a list of dictionaries from web search.")
        web_results = Document(page_content="No valid results found.")

    web_results = Document(page_content=web_results)
    documents.append(web_results)

    return {"documents": documents, "question": question}


def decide_to_generate_edge(state):
    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"
