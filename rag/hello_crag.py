from pathlib import Path
from langchain_community.document_loaders import MHTMLLoader
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os
os.environ["USER_AGENT"] = "LangChain/RAG-Application"


files = [
    "data/LLM Powered Autonomous Agents.mhtml",
]
print("---Load documents---")
docs = [MHTMLLoader(Path(file)).load() for file in files]
docs_list = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

print("---Embeddings model---")
# https://ollama.com/blog/embedding-models
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)
print("---vectorstore with chroma---")
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="nomic-chroma-vectorstore",
    embedding=embeddings,
)
print("---RETRIEVE---")
retriever = vectorstore.as_retriever()
print(retriever.invoke("What is an agent?"))
