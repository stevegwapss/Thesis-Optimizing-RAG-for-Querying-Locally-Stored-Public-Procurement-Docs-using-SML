#!/usr/bin/env python3
"""
pdf2jsonl_text.py

Convert text-based PDFs into a JSONL dataset for LLaMA fine-tuning.

Usage:
    python pdf2jsonl_text.py --pdf_dir my_pdfs --output dataset.jsonl --chunk_words 300
"""

import argparse, os, glob, json
from pypdf import PdfReader

def extract_text(pdf_path: str) -> str:
    try:
        reader = PdfReader(pdf_path)
        return "\n".join([p.extract_text() or "" for p in reader.pages])
    except Exception as e:
        print(f"[WARN] Could not read {pdf_path}: {e}")
        return ""

def chunk_text(text: str, max_words: int = 300):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i+max_words])

def main(pdf_dir, output_file, chunk_words):
    samples = []
    for pdf_file in glob.glob(os.path.join(pdf_dir, "*.pdf")):
        text = extract_text(pdf_file)
        if not text.strip():
            continue
        for chunk in chunk_text(text, chunk_words):
            samples.append({"text": chunk})

    with open(output_file, "w", encoding="utf-8") as f:
        for row in samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(samples)} samples to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", type=str, required=True, help="Directory of PDFs")
    parser.add_argument("--output", type=str, default="dataset.jsonl", help="Output JSONL file")
    parser.add_argument("--chunk_words", type=int, default=300, help="Max words per chunk")
    args = parser.parse_args()
    main(args.pdf_dir, args.output, args.chunk_words)
