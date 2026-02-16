from typing import List, Optional

from src.state.rag_state import RAGState
from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

class RAGNodes:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None  # type: ignore

    def retrieve_docs(self, state: RAGState) -> RAGState:
        docs = self.retriever.invoke(state["question"])
        return {
            "question": state["question"],
            "retrieved_docs": docs,
            "answer": state.get("answer", "")
        }

    def _build_tools(self):
        def retriever_tool_fn(query:str)->str:
            docs:List[Document] = self.retriever.invoke(query)
            if not docs:
                return "No documents found"
            merged = []
            for i, d in enumerate(docs[:8], start=1):
                meta = d.metadata if hasattr(d, "metadata") else {}
                title = meta.get("title") or meta.get("source") or f"Document_{i}"
                merged.append(f"Document {i}: {title}\n{d.page_content}")
            return "\n\n".join(merged)

        retriever_tool = Tool(
            name = "retriever",
            description= "Fetch passages from indexed vectorstore",
            func = retriever_tool_fn
        )

        # Create Wikipedia tool using wikipedia library directly
        def wikipedia_search(query: str) -> str:
            try:
                import wikipedia
                return wikipedia.summary(query, sentences=3)
            except Exception as e:
                return f"Wikipedia search failed: {str(e)}"
        
        wikipedia_tool = Tool(
            name="wikipedia",
            description="Search wikipedia for general knowledge",
            func=wikipedia_search
        )

        return [retriever_tool, wikipedia_tool]

    def _build_agent(self):
        tools = self._build_tools()
        self._agent = create_react_agent(self.llm, tools=tools)

    def generate_answer(self, state: RAGState) -> RAGState:
        if self._agent is None:
            self._build_agent()

        result = self._agent.invoke({"messages": [HumanMessage(content=state["question"])]})  # type: ignore

        messages = result.get("messages", [])
        answer: Optional[str] = None
        if messages:
            answer_msg = messages[-1]
            answer = getattr(answer_msg, "content", None)

        return {
            "question": state["question"],
            "retrieved_docs": state.get("retrieved_docs", []),
            "answer": answer or "Could not generate answer"
        }