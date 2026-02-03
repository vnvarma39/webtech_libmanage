import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

@st.cache_resource(show_spinner=False)
def load_section_summarizer(model_path_or_name="facebook/bart-large-cnn"):
    return pipeline("summarization", model=model_path_or_name)

@st.cache_resource(show_spinner=False)
def load_global_model(model_name="google/bigbird-pegasus-large-arxiv"):
    tok = AutoTokenizer.from_pretrained(model_name)
    mod = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tok, mod

def summarize_chunks(chunks, section_summarizer, max_len=220, min_len=60):
    summaries = []
    for ch in chunks:
        out = section_summarizer(ch, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']
        summaries.append(out)
    return " ".join(summaries)

def long_context_summary(text, tok, mod, max_input=4096, max_out=512):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=max_input)
    ids = mod.generate(inputs["input_ids"], num_beams=4, max_length=max_out)
    return tok.decode(ids[0], skip_special_tokens=True)