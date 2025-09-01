#!/usr/bin/env python3
"""
separate_pdfs.py

Sort PDFs into:
- text_pdfs/  (can be read with pypdf)
- ocr_pdfs/   (require OCR)

Usage:
    python separate_pdfs.py --pdf_dir ./my_pdfs --out_dir ./sorted_pdfs
"""

import os
import glob
import shutil
import argparse
from pypdf import PdfReader

def has_extractable_text(pdf_path: str) -> bool:
    """Check if a PDF has extractable text using pypdf."""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            if page.extract_text() and page.extract_text().strip():
                return True
        return False
    except Exception as e:
        print(f"[WARN] Failed to parse {pdf_path}: {e}")
        return False

def main(pdf_dir, out_dir):
    text_dir = os.path.join(out_dir, "text_pdfs")
    ocr_dir = os.path.join(out_dir, "ocr_pdfs")

    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(ocr_dir, exist_ok=True)

    for pdf_file in glob.glob(os.path.join(pdf_dir, "*.pdf")):
        print(f"📄 Checking {pdf_file} ...")
        if has_extractable_text(pdf_file):
            print(f"   ✅ Text found → {text_dir}")
            shutil.copy2(pdf_file, text_dir)
        else:
            print(f"   🔍 No text found → {ocr_dir}")
            shutil.copy2(pdf_file, ocr_dir)

    print(f"\n✅ Done! Separated PDFs into:\n{text_dir}\n{ocr_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", type=str, required=True, help="Folder with all PDFs")
    parser.add_argument("--out_dir", type=str, default="./sorted_pdfs", help="Output base folder")
    args = parser.parse_args()
    main(args.pdf_dir, args.out_dir)
