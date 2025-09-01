import os
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from pypdf import PdfReader

print("🔍 Dependency Test Starting...\n")

# --- 1. Check Tesseract ---
try:
    if os.name == "nt":  # Windows
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    print("✅ Tesseract version:", pytesseract.get_tesseract_version())
except Exception as e:
    print("❌ Tesseract check failed:", e)

# --- 2. Check Pillow ---
try:
    img = Image.new("RGB", (100, 50), color="white")
    img.save("test_image.png")
    print("✅ Pillow works: test_image.png created")
except Exception as e:
    print("❌ Pillow check failed:", e)

# --- 3. Check Poppler + pdf2image ---
sample_pdf = "sample.pdf"
try:
    if not os.path.exists(sample_pdf):
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            c = canvas.Canvas(sample_pdf, pagesize=letter)
            c.drawString(100, 750, "Hello PDF")
            c.save()
            print("✅ Created sample.pdf using reportlab")
        except ImportError:
            print("❌ reportlab not installed → run `pip install reportlab`")
            sample_pdf = None

    if sample_pdf:
        images = convert_from_path(sample_pdf, dpi=100)
        images[0].save("test_pdf_page.png", "PNG")
        print("✅ Poppler + pdf2image works: test_pdf_page.png created")
except Exception as e:
    print("❌ pdf2image/Poppler check failed:", e)

# --- 4. Check PyPDF ---
if sample_pdf and os.path.exists(sample_pdf):
    try:
        reader = PdfReader(sample_pdf)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        print("✅ PyPDF works, extracted text:", text.strip())
    except Exception as e:
        print("❌ PyPDF check failed:", e)
else:
    print("⚠️ Skipping PyPDF check (no sample.pdf found)")

# --- 5. OCR test ---
if os.path.exists("test_pdf_page.png"):
    try:
        ocr_result = pytesseract.image_to_string(Image.open("test_pdf_page.png"))
        print("✅ OCR works, recognized text:", repr(ocr_result.strip()))
    except Exception as e:
        print("❌ OCR test failed:", e)
else:
    print("⚠️ Skipping OCR test (no test_pdf_page.png found)")

print("\n🎉 Dependency Test Completed")
