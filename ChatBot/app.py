import streamlit as st
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

st.set_page_config(page_title="LangChain Memory Chatbot", page_icon="🤖")

st.title("🤖 LangChain Chatbot with Memory")
st.write("Week 2 Deliverable: Streamlit + Ollama + LangChain + SQLite Memory")


# Sidebar settings
st.sidebar.header("⚙️ Settings")

base_url = st.sidebar.text_input("Ollama Base URL", "http://localhost:11434")

model = st.sidebar.selectbox(
    "Choose Model",
    ["llama3", "llama2", "mistral", "gemma"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.write("📌 Chat history is stored in SQLite database.")


# User ID
user_id = st.text_input("👤 Enter your User ID", "dhanush")


# Renamed DB file
DB_FILE = "dhanush_chat_history.db"


def get_session_history(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection=f"sqlite:///{DB_FILE}"
    )


# Initialize UI chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Clear conversation
if st.button("🆕 Start New Conversation"):
    history = get_session_history(user_id)
    history.clear()
    st.session_state.chat_history = []
    st.success("Conversation cleared successfully ✅")


# Display messages in UI
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Setup Ollama LLM
llm = ChatOllama(base_url=base_url, model=model)

system_prompt = """
You are a friendly AI assistant.
Speak naturally like a human.
Answer clearly and step-by-step if user asks technical questions.
"""

system = SystemMessagePromptTemplate.from_template(system_prompt)
human = HumanMessagePromptTemplate.from_template("{input}")

prompt_template = ChatPromptTemplate.from_messages([
    system,
    MessagesPlaceholder(variable_name="history"),
    human
])

chain = prompt_template | llm | StrOutputParser()

runnable_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)


def chat_with_llm(session_id, user_input):
    for chunk in runnable_with_history.stream(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    ):
        yield chunk


# Chat input
user_prompt = st.chat_input("Ask something...")

if user_prompt:
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(chat_with_llm(user_id, user_prompt))

    st.session_state.chat_history.append({"role": "assistant", "content": response})

