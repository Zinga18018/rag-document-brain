"""
streamlit UI for the RAG pipeline.
upload docs, ask questions, get grounded answers.
"""

import streamlit as st
from core import RAGPipeline, RAGConfig

st.set_page_config(page_title="Document Q&A", layout="wide")


@st.cache_resource
def load_pipeline():
    return RAGPipeline(RAGConfig())


st.title("Document Q&A")
st.caption("retrieval-augmented generation with Sentence-Transformers + ChromaDB")

pipeline = load_pipeline()

tab_ingest, tab_ask = st.tabs(["upload documents", "ask questions"])

with tab_ingest:
    st.subheader("add a document")

    uploaded = st.file_uploader("drop a file here", type=["txt", "md", "py", "csv", "pdf"])
    if uploaded:
        content = uploaded.read()
        with st.spinner("chunking and embedding..."):
            result = pipeline.ingest_bytes(content, uploaded.name)
        st.success(
            f"ingested **{uploaded.name}** — "
            f"{result['chunks_created']} chunks in {result['ingest_ms']:.0f}ms"
        )

    st.divider()

    with st.expander("or paste raw text"):
        source_name = st.text_input("source name", value="my-notes")
        raw_text = st.text_area("text content", height=200)
        if st.button("ingest text") and raw_text.strip():
            result = pipeline.ingest_text(raw_text, source_name)
            st.success(f"ingested {result['chunks_created']} chunks")

    # sidebar stats
    status = pipeline.status()
    st.sidebar.metric("total chunks", status["total_chunks"])
    st.sidebar.metric("sources", len(status["sources"]))
    if status["sources"]:
        st.sidebar.write("**ingested files:**")
        for s in status["sources"]:
            st.sidebar.write(f"- {s}")

with tab_ask:
    st.subheader("search your documents")

    question = st.text_input(
        "what do you want to know?",
        placeholder="how does the chunking algorithm work?",
    )
    top_k = st.slider("results to return", 1, 10, 5)

    if st.button("search", type="primary") and question.strip():
        with st.spinner("searching..."):
            results = pipeline.query(question, top_k=top_k)

        st.write(f"found {len(results['sources'])} relevant chunks in {results['query_ms']:.0f}ms")

        for i, hit in enumerate(results["sources"]):
            score_pct = hit["similarity"] * 100
            st.markdown(f"**chunk {i+1}** from `{hit['filename']}` — {score_pct:.1f}% match")
            st.text(hit["text"][:500])
            st.divider()
