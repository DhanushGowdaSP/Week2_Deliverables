# RAG Q&A System

A Retrieval-Augmented Generation (RAG) system that answers questions from your documents using LangChain, LangGraph, and Streamlit.

## Features

- 📄 **Document Processing**: Load and process PDF documents
- 🌐 **Web Content**: Fetch and analyze content from URLs
- 🔍 **Smart Retrieval**: Vector-based document search using FAISS
- 🤖 **Intelligent Answers**: AI-powered question answering
- 💬 **Interactive UI**: User-friendly Streamlit interface
- 🔧 **ReAct Agent**: Enhanced reasoning with Wikipedia integration

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd RAG
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. Open your browser to http://localhost:8501

3. Click **"Initialize System"** in the sidebar to load documents

4. Enter your question in the text box and click **"Get Answer"**

## Configuration

Edit the `.env` file to configure:

- `LLM_PROVIDER`: Choose your LLM provider (ollama, groq, openai)
- `LLM_MODEL`: Specify the model to use
- API keys (if required by your provider)

## Project Structure

```
RAG/
├── src/
│   ├── config/          # Configuration and LLM setup
│   ├── doc_ingestion/   # Document loading and processing
│   ├── vector_store/    # FAISS vector store management
│   ├── nodes/           # LangGraph nodes (retrieval, generation)
│   ├── graph_builder/   # LangGraph workflow orchestration
│   └── state/           # State management
├── data/                # Place your PDF files here
├── app.py              # Streamlit application
├── requirements.txt    # Python dependencies
└── .env               # Environment configuration
```

## How It Works

1. **Document Ingestion**: PDFs and web content are loaded and split into chunks
2. **Vector Storage**: Document chunks are embedded and stored in FAISS
3. **Question Processing**: User questions are processed by the LangGraph workflow
4. **Retrieval**: Relevant document chunks are retrieved using vector similarity
5. **Answer Generation**: The LLM generates answers based on retrieved context
6. **ReAct Agent**: Uses tools (retriever, Wikipedia) to provide comprehensive answers

## Adding Documents

### PDFs
Place your PDF files in the `data/` directory. They will be automatically loaded when you initialize the system.

### URLs
Edit `src/config/config.py` to add URLs to the `DEFAULT_URLS` list:
```python
DEFAULT_URLS = [
    "https://example.com/article1",
    "https://example.com/article2"
]
```

## Technologies Used

- **LangChain**: Framework for LLM applications
- **LangGraph**: Workflow orchestration
- **FAISS**: Vector similarity search
- **Streamlit**: Web interface
- **PyPDF**: PDF processing
- **BeautifulSoup**: Web scraping

## License

MIT