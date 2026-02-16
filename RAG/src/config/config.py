import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # ollama, openai, groq
    LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    DEFAULT_URLS = [
        "https://www.ibm.com/think/topics/agentic-ai"
    ]

    @classmethod
    def get_llm(cls):
        """Get LLM based on provider setting"""
        provider = cls.LLM_PROVIDER.lower()
        
        if provider == "ollama":
            # Use Ollama (free, local)
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=cls.LLM_MODEL,
                temperature=0
            )
        elif provider == "groq":
            # Use Groq (free API with rate limits)
            from langchain_groq import ChatGroq
            groq_api_key = os.getenv("GROQ_API_KEY")
            return ChatGroq(
                model=cls.LLM_MODEL,
                temperature=0,
                api_key=groq_api_key
            )
        elif provider == "openai":
            # Use OpenAI (paid)
            from langchain_openai import ChatOpenAI
            if cls.OPENAI_API_KEY:
                os.environ["OPENAI_API_KEY"] = cls.OPENAI_API_KEY
            return ChatOpenAI(model=cls.LLM_MODEL, temperature=0)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")