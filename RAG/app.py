import streamlit as st
from src.config.config import Config
from src.doc_ingestion.doc_processor import DocumentProcessor
from src.vector_store.vstore import VectorStore
from src.graph_builder.graph_build import GraphBuilder

st.set_page_config(page_title="RAG Q&A System", page_icon="🤖", layout="wide")

st.title("🤖 RAG Q&A System")
st.markdown("Ask questions about your documents!")

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.graph = None

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Document sources
    st.subheader("Document Sources")
    use_pdfs = st.checkbox("Load PDFs from data/ directory", value=True)
    use_urls = st.checkbox("Load URLs", value=True)
    
    if st.button("Initialize System"):
        with st.spinner("Initializing RAG system..."):
            try:
                # Initialize components
                doc_processor = DocumentProcessor()
                vector_store = VectorStore()
                llm = Config.get_llm()
                
                # Process documents
                documents = []
                if use_pdfs:
                    pdf_docs = doc_processor.load_from_directory("data")
                    documents.extend(pdf_docs)
                    st.success(f"Loaded {len(pdf_docs)} PDF chunks")
                
                if use_urls:
                    urls = Config.DEFAULT_URLS
                    url_docs = doc_processor.load_urls(urls)
                    documents.extend(url_docs)
                    st.success(f"Loaded {len(url_docs)} URL chunks")
                
                if not documents:
                    st.error("No documents loaded. Please enable at least one source.")
                else:
                    # Create vector store
                    vector_store.create_retriever(documents)
                    retriever = vector_store.get_retriever()
                    
                    # Build graph
                    graph_builder = GraphBuilder(retriever, llm)
                    st.session_state.graph = graph_builder
                    st.session_state.initialized = True
                    
                    st.success(f"✅ System initialized with {len(documents)} document chunks!")
            except Exception as e:
                st.error(f"Error initializing system: {str(e)}")

# Main content area
if st.session_state.initialized:
    st.success("System is ready! Ask your questions below.")
    
    # Question input
    question = st.text_input("Enter your question:", placeholder="What would you like to know?")
    
    if st.button("Get Answer", type="primary"):
        if question:
            with st.spinner("Generating answer..."):
                try:
                    result = st.session_state.graph.run(question)
                    
                    # Display answer
                    st.subheader("Answer:")
                    st.write(result.get("answer", "No answer generated"))
                    
                    # Display retrieved documents
                    with st.expander("View Retrieved Documents"):
                        docs = result.get("retrieved_docs", [])
                        if docs:
                            for i, doc in enumerate(docs, 1):
                                st.markdown(f"**Document {i}:**")
                                st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                                st.markdown("---")
                        else:
                            st.info("No documents retrieved")
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
        else:
            st.warning("Please enter a question")
else:
    st.info("👈 Please initialize the system using the sidebar")
    st.markdown("""
    ### How to use:
    1. Click **Initialize System** in the sidebar
    2. Wait for documents to load
    3. Enter your question in the text box
    4. Click **Get Answer** to receive a response
    
    The system uses:
    - 📄 PDF documents from the `data/` directory
    - 🌐 Web content from configured URLs
    - 🤖 OpenAI GPT-4 for intelligent responses
    - 🔍 Vector search for relevant context
    """)

# Made with Bob
