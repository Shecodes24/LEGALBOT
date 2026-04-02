import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from transformers import pipeline
import dotenv
import langchain_community
import torch
import transformers


print("langchain_community version:", langchain_community.__version__)
print("torch version:", torch.__version__)
print("transformers version:", transformers.__version__)

# --------------------------------------------------
# Device
# --------------------------------------------------
device = 0 if torch.cuda.is_available() else -1
print("Using device:", "GPU" if device == 0 else "CPU")

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=device
)


PDF_PATH = "case.pdf"  

loader = PyPDFLoader(PDF_PATH)
docs = loader.load()


full_text = "\n".join(doc.page_content for doc in docs)

print("✅ PDF loaded successfully")
print(full_text)
print(f"Total pages: {len(docs)}")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200
)

chunks = text_splitter.split_text(full_text)
print(f"✅ Text split into {len(chunks)} chunks")

partial_summaries = []

for idx, case in enumerate(chunks):
    summary = summarizer(
        case,
        max_length=200,
        min_length=60,
        do_sample=False
    )
    
    response = summary[0]["summary_text"]
    partial_summaries.append(response)
    print(f"Summarized chunk {idx + 1}/{len(chunks)}")

final_summary = f"""
SUMMARIES:
{''.join(partial_summaries)}
"""

print("\n" + "="*60)
print("📄 FINAL LEGAL DOCUMENT SUMMARY")
print("="*60)
print(final_summary)
