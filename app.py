import os
import glob
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="💼", layout="centered")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
LANGCHAIN_API_KEY = st.secrets.get("LANGCHAIN_API_KEY", "")

os.environ["GROQ_API_KEY"] = GROQ_API_KEY
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "zyro-rag-challenge"

PDF_DIR = "data"
SCORE_THRESHOLD = 0.45

SYSTEM_PROMPT = """You are the Zyro Dynamics HR Help Desk Assistant.
Rules:
1. Answer ONLY using the provided context from Zyro Dynamics HR policy documents.
2. Be specific and cite exact numbers, durations, and policy names when present.
3. If the context does NOT contain the answer, or the question is unrelated to Zyro Dynamics HR policies, respond EXACTLY with:
"I can only answer HR-related questions from Zyro Dynamics policy documents."
4. Do not make up information. Do not use outside knowledge.
5. Keep answers clear, complete, and professional - 2 to 5 sentences.
Context:
{context}
"""

@st.cache_resource(show_spinner="Loading HR policy documents...")
def build_rag_pipeline():
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    if not pdf_files:
        st.error(f"No PDFs found in '{PDF_DIR}/'.")
        st.stop()

    docs = []
    for f in pdf_files:
        loader = PyPDFLoader(f)
        pages = loader.load()
        for p in pages:
            p.metadata["source"] = os.path.basename(f)
        docs.extend(pages)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5}
    )

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])

    def format_docs(docs_):
        return "\n\n".join(f"[{d.metadata.get('source','')}] {d.page_content}" for d in docs_)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return vectorstore, rag_chain

vectorstore, rag_chain = build_rag_pipeline()

def answer_query(question: str):
    docs_with_scores = vectorstore.similarity_search_with_relevance_scores(question, k=5)
    if not docs_with_scores or all(score < SCORE_THRESHOLD for _, score in docs_with_scores):
        return "I can only answer HR-related questions from Zyro Dynamics policy documents.", []
    answer = rag_chain.invoke(question)
    sources = sorted(set(d.metadata.get("source", "Unknown") for d, _ in docs_with_scores))
    return answer, sources

st.title("💼 Zyro Dynamics HR Help Desk")
st.caption("Ask me anything about leave, WFH, conduct, benefits, onboarding and more.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 Sources"):
                for s in msg["sources"]:
                    st.write(f"- {s}")

user_input = st.chat_input("Ask your HR question...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = answer_query(user_input)
            st.markdown(answer)
            if sources:
                with st.expander("📄 Sources"):
                    for s in sources:
                        st.write(f"- {s}")
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

with st.sidebar:
    st.header("ℹ️ About")
    st.write("Answers HR questions using Zyro Dynamics official policy documents.")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()
