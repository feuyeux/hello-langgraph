import os
os.environ["USER_AGENT"] = "LangChain/RAG-Application"
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import MHTMLLoader
from pathlib import Path

def vectorstore_retriever(collection_name):
    # Load documents from local MHTML files
    # urls = [
    #     "https://lilianweng.github.io/posts/2023-06-23-agent/",
    #     "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    #     "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    # ]
    files = [
        "data/LLM Powered Autonomous Agents.mhtml",
        "data/Prompt Engineering.mhtml",
        "data/Adversarial Attacks on LLMs.mhtml",
    ]
    docs = [MHTMLLoader(Path(file)).load() for file in files]
    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Embeddings model
    # https://ollama.com/blog/embedding-models
    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large"
    )
    
    # embedding=NomicEmbeddings(
    #     model="nomic-embed-text-v1.5",
    #     inference_mode="local",
    # )

    # Add to vectorDB
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name=collection_name,
        persist_directory=".",
        embedding=embeddings,    
    )
    
    # vectorstore = SKLearnVectorStore.from_documents(
    #     documents=doc_splits,
    #     embedding=embedding,
    # )
    
    return vectorstore.as_retriever()

retriever = vectorstore_retriever("agentic-rag-chroma")
# 
documents = retriever.invoke("What does Lilian Weng say about the types of agent memory?")
print("",documents[0].page_content)