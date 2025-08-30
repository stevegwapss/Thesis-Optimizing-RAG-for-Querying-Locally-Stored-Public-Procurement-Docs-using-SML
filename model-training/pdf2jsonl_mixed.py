#!/usr/bin/env python3
"""
pdf2jsonl_mixed.py

Convert a folder of PDFs (scanned or text-based) into a JSONL dataset
ready for LLaMA fine-tuning.

- First tries to extract text using pypdf.
- If no text is found, falls back to OCR using Tesseract.

Requirements:
    pip install pypdf pytesseract pdf2image pillow

Also install Tesseract OCR engine:
    - Ubuntu/Debian: sudo apt install tesseract-ocr
    - Mac (brew): brew install tesseract
    - Windows: https://github.com/UB-Mannheim/tesseract/wiki

Usage:
    python pdf2jsonl_mixed.py --pdf_dir ./my_pdfs --output dataset.jsonl --chunk_words 300
"""

import argparse, os, glob, json
from pypdf import PdfReader
import pytesseract
from pdf2image import convert_from_path

def extract_text_pypdf(pdf_path: str) -> str:
    """Try extracting text with pypdf (works for normal PDFs)."""
    try:
        reader = PdfReader(pdf_path)
        return "\n".join([p.extract_text() or "" for p in reader.pages]).strip()
    except Exception as e:
        print(f"[WARN] pypdf failed for {pdf_path}: {e}")
        return ""

def extract_text_ocr(pdf_path: str) -> str:
    """Fallback: OCR for scanned PDFs."""
    try:
        pages = convert_from_path(pdf_path, dpi=200)
        text = ""
        for page in pages:
            text += pytesseract.image_to_string(page, lang="eng") + "\n"
        return text.strip()
    except Exception as e:
        print(f"[WARN] OCR failed for {pdf_path}: {e}")
        return ""

def chunk_text(text: str, max_words: int = 300):
    """Split long text into chunks of N words."""
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i+max_words])

def main(pdf_dir, output_file, chunk_words):
    samples = []
    for pdf_file in glob.glob(os.path.join(pdf_dir, "*.pdf")):
        print(f"📄 Processing {pdf_file} ...")
        text = extract_text_pypdf(pdf_file)
        if not text:
            print(f"   → No text found, running OCR...")
            text = extract_text_ocr(pdf_file)

        if not text:
            print(f"   ⚠️ Skipping {pdf_file}, no text extracted.")
            continue

        for chunk in chunk_text(text, chunk_words):
            samples.append({"text": chunk})

    with open(output_file, "w", encoding="utf-8") as f:
        for row in samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n✅ Finished! Extracted {len(samples)} chunks into {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", type=str, required=True, help="Folder with all PDFs")
    parser.add_argument("--output", type=str, default="dataset.jsonl", help="Output JSONL file")
    parser.add_argument("--chunk_words", type=int, default=300, help="Max words per chunk")
    args = parser.parse_args()
    main(args.pdf_dir, args.output, args.chunk_words)
