from pdfminer.high_level import extract_text
from docx import Document

def extract_text_from_pdf(path):
    return extract_text(path)

def extract_text_from_docx(path):
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs])