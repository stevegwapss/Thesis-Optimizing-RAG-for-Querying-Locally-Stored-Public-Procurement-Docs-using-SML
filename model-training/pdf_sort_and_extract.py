#!/usr/bin/env python3
"""
pdf_sort_and_extract.py

1. Separates PDFs into text-based vs scanned (OCR).
2. Extracts text from each using the appropriate method.
3. Outputs a combined JSONL dataset ready for fine-tuning.

Requirements:
    pip install pypdf pytesseract pdf2image pillow
    Install poppler + tesseract (system)

Usage:
    python pdf_sort_and_extract.py --pdf_dir ./my_pdfs --out_dir ./sorted_pdfs --output dataset.jsonl --chunk_words 300
"""

import os, glob, json, argparse, shutil
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_text_pypdf(pdf_path: str) -> str:
    """Try extracting text using PyPDF (fast for text PDFs)."""
    try:
        reader = PdfReader(pdf_path)
        return "\n".join([p.extract_text() or "" for p in reader.pages]).strip()
    except Exception:
        return ""

def extract_text_ocr(pdf_path: str) -> str:
    """Fallback: OCR extraction for scanned PDFs."""
    text = ""
    try:
        pages = convert_from_path(pdf_path, dpi=200)
        for page in pages:
            text += pytesseract.image_to_string(page, lang="eng") + "\n"
    except Exception as e:
        print(f"[WARN] OCR failed for {pdf_path}: {e}")
    return text.strip()

def chunk_text(text: str, max_words: int = 300):
    """Split text into word chunks."""
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i+max_words])

def main(pdf_dir, out_dir, output_file, chunk_words):
    os.makedirs(out_dir, exist_ok=True)
    text_dir = os.path.join(out_dir, "text_pdfs")
    ocr_dir = os.path.join(out_dir, "ocr_pdfs")
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(ocr_dir, exist_ok=True)

    dataset = []

    for pdf_file in glob.glob(os.path.join(pdf_dir, "*.pdf")):
        print(f"📄 Checking {pdf_file} ...")
        text = extract_text_pypdf(pdf_file)

        if text:
            dest = os.path.join(text_dir, os.path.basename(pdf_file))
            shutil.copy(pdf_file, dest)
            print(f"   ✅ Text PDF → {dest}")
            extractor = extract_text_pypdf
        else:
            dest = os.path.join(ocr_dir, os.path.basename(pdf_file))
            shutil.copy(pdf_file, dest)
            print(f"   🖼️ Scanned PDF (OCR) → {dest}")
            extractor = extract_text_ocr

        # Extract text for dataset
        extracted_text = extractor(pdf_file)
        if not extracted_text:
            print(f"   ⚠️ Skipping {pdf_file}, no text extracted.")
            continue

        for chunk in chunk_text(extracted_text, chunk_words):
            dataset.append({"text": chunk})

    with open(output_file, "w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n✅ Finished! Extracted {len(dataset)} chunks into {output_file}")
    print(f"   Sorted PDFs saved under: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", type=str, required=True, help="Folder with PDFs")
    parser.add_argument("--out_dir", type=str, default="sorted_pdfs", help="Output folder for sorted PDFs")
    parser.add_argument("--output", type=str, default="dataset.jsonl", help="Output JSONL dataset")
    parser.add_argument("--chunk_words", type=int, default=300, help="Max words per chunk")
    args = parser.parse_args()
    main(args.pdf_dir, args.out_dir, args.output, args.chunk_words)
