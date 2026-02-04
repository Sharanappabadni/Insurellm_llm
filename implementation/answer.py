# answer.py
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv(override=True)

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
DB_NAME = str(Path(__file__).parent / "vector_db")
COLLECTION_NAME = "docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Insurellm Assistant",
    page_icon="🏢",
    layout="wide",
)

st.title("🏢 Insurellm Knowledge Assistant")


# ------------------------------------------------------------------
# Embeddings + Vector Store
# ------------------------------------------------------------------
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = Chroma(
        persist_directory=DB_NAME,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    try:
        count = vectorstore._collection.count()
        st.sidebar.info(f"Vector DB docs: {count}")
    except Exception as e:
        st.sidebar.error(f"Chroma error: {e}")

    return vectorstore


vectorstore = load_vectorstore()

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": TOP_K},
)


# ------------------------------------------------------------------
# LLM (Gemini via OpenAI-compatible endpoint)
# ------------------------------------------------------------------
llm = ChatOpenAI(
    model="gemini-3-flash-preview",
    openai_api_key=os.getenv("GOOGLE_API_KEY"),
    openai_api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    temperature=0.2,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def format_docs(docs):
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "Unknown")
        parts.append(f"[Source {i}: {src}]\n{d.page_content}")
    return "\n\n".join(parts)


# ------------------------------------------------------------------
# Prompt (STRICT grounding)
# ------------------------------------------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert assistant for Insurellm.\n"
            "Answer ONLY using the provided context.\n"
            "If the answer is not present, say 'I don't know'.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


# ------------------------------------------------------------------
# RAG Chain – standard LCEL pattern
# ------------------------------------------------------------------
# Input to the chain is just the question string.
# RunnablePassthrough() forwards that string as "question".
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)  # [web:33][web:36]


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
question = st.text_input(
    "Ask a question about Insurellm:",
    placeholder="e.g. What is the cancellation policy?",
)

if question:
    with st.spinner("Searching knowledge base..."):
        # Debug: directly show retrieved docs
        retrieved_docs = retriever.invoke(question)

        st.subheader("📚 Retrieved Context (raw)")
        if not retrieved_docs:
            st.warning("No documents retrieved – check ingestion and paths.")
        else:
            for i, doc in enumerate(retrieved_docs, 1):
                with st.expander(
                    f"Source {i}: {doc.metadata.get('source', 'Unknown')}"
                ):
                    st.write(doc.page_content)

        # Run full RAG chain
        answer = rag_chain.invoke(question)

    st.subheader("✅ Answer")
    st.write(answer)
