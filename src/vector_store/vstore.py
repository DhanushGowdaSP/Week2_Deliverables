from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
import os

class VectorStore:
    def __init__(self):
        # Use free Ollama embeddings instead of OpenAI
        embedding_provider = os.getenv("LLM_PROVIDER", "ollama")
        
        if embedding_provider == "ollama":
            self.embedding = OllamaEmbeddings(model="nomic-embed-text")
        else:
            # Fallback to OpenAI if specified
            from langchain_openai import OpenAIEmbeddings
            self.embedding = OpenAIEmbeddings()
            
        self.vector_store = None
        self.retriever = None

    def create_retriever(self, documents : List[Document]):
        self.vector_store = FAISS.from_documents(documents, self.embedding)
        self.retriever = self.vector_store.as_retriever()

    def get_retriever(self):
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.retriever

    def retrieve(self, query : str, k : int = 4):
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.retriever.invoke(query)