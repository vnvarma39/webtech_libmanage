import re

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, max_chars=2500):
    # chunk by paragraphs roughly respecting max_chars
    paras = re.split(r'\n+', text)
    chunks, current = [], ""
    for p in paras:
        if len(current) + len(p) + 1 <= max_chars:
            current += (p + " ")
        else:
            if current.strip():
                chunks.append(current.strip())
            current = p + " "
    if current.strip():
        chunks.append(current.strip())
    return chunks