from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from src.config.config import Config
import os

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """Load and split PDF documents"""
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
    def load_urls(self, urls: List[str]) -> List[Document]:
        """Load and split web documents"""
        loader = WebBaseLoader(urls)
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
    def load_from_directory(self, directory: str) -> List[Document]:
        """Load all PDF files from a directory"""
        all_documents = []
        for filename in os.listdir(directory):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(directory, filename)
                documents = self.load_pdf(pdf_path)
                all_documents.extend(documents)
        return all_documents
    
    def process_documents(self, pdf_dir: Optional[str] = None, urls: Optional[List[str]] = None) -> List[Document]:
        """Process documents from PDFs and/or URLs"""
        all_documents = []
        
        if pdf_dir and os.path.exists(pdf_dir):
            pdf_docs = self.load_from_directory(pdf_dir)
            all_documents.extend(pdf_docs)
        
        if urls:
            url_docs = self.load_urls(urls)
            all_documents.extend(url_docs)
        
        return all_documents

# Made with Bob
