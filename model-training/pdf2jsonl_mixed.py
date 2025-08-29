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
    """Try extracting text with pypdf (works for digital PDFs)."""
    text_parts = []
    try:
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            except Exception as e:
                print(f"[WARN] Failed extracting page {i} in {pdf_path}: {e}")
        return "\n".join(text_parts).strip()
    except Exception as e:
        print(f"[WARN] pypdf failed for {pdf_path}: {e}")
        return ""

def extract_text_ocr(pdf_path: str) -> str:
    """Fallback: OCR for scanned PDFs."""
    try:
        pages = convert_from_path(pdf_path, dpi=200)
        text_parts = []
        for i, page in enumerate(pages):
            page_text = pytesseract.image_to_string(page, lang="eng")
            if page_text.strip():
                text_parts.append(page_text)
        return "\n".join(text_parts).strip()
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
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    print(f"📂 Found {len(pdf_files)} PDF(s) in {pdf_dir}")

    for pdf_file in pdf_files:
        print(f"\n📄 Processing {os.path.basename(pdf_file)} ...")
        text = extract_text_pypdf(pdf_file)

        if not text:
            print(f"   → No text found with pypdf, running OCR...")
            text = extract_text_ocr(pdf_file)

        if not text:
            print(f"   ⚠️ Skipping {os.path.basename(pdf_file)} (no text extracted).")
            continue

        chunks = list(chunk_text(text, chunk_words))
        samples.extend({"text": chunk} for chunk in chunks)
        print(f"   ✅ Extracted {len(chunks)} chunks")

    with open(output_file, "w", encoding="utf-8") as f:
        for row in samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n🎉 Finished! Extracted {len(samples)} chunks into {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", type=str, required=True, help="Folder with all PDFs")
    parser.add_argument("--output", type=str, default="dataset.jsonl", help="Output JSONL file")
    parser.add_argument("--chunk_words", type=int, default=300, help="Max words per chunk")
    args = parser.parse_args()
    main(args.pdf_dir, args.output, args.chunk_words)
