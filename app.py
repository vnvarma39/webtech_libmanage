import streamlit as st
from ingest import extract_text_from_pdf, extract_text_from_docx
from preprocess import clean_text, chunk_text
from summarizer import load_section_summarizer, load_global_model, summarize_chunks, long_context_summary

st.set_page_config(page_title="🧠 The Alchemist Abstractor", layout="wide")
st.title("🧠 The Alchemist Abstractor")
st.caption("Upload a PDF or DOCX to generate an abstractive summary.")

model_choice = st.sidebar.selectbox(
    "Section-level model",
    ["./models/fine_tuned_model", "facebook/bart-large-cnn", "sshleifer/distilbart-cnn-12-6"],
    index=0
)

uploaded = st.file_uploader("Upload PDF/DOCX", type=["pdf", "docx"])
if uploaded:
    with open("tempfile", "wb") as f:
        f.write(uploaded.getbuffer())

    text = extract_text_from_pdf("tempfile") if uploaded.name.endswith(".pdf") else extract_text_from_docx("tempfile")
    text = clean_text(text)
    chunks = chunk_text(text, max_chars=2500)
    st.info(f"Document split into {len(chunks)} chunks.")

    section_summarizer = load_section_summarizer(model_choice)
    st.info("Generating section-level summaries…")
    section_summary = summarize_chunks(chunks, section_summarizer)

    st.info("Generating final long-context summary…")
    tok, mod = load_global_model("google/bigbird-pegasus-large-arxiv")
    final_summary = long_context_summary(section_summary, tok, mod)

    st.subheader("📄 Final Abstractive Summary")
    st.write(final_summary)
    st.download_button("⬇️ Download Summary", final_summary, file_name="summary.txt")