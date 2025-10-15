"""
Test PDF extraction to verify table data is being extracted correctly
"""
import os

pdf_path = "uploads/UPDATED_-APP_2013.pdf"

print("=" * 80)
print("🔍 Testing PDF Extraction Methods")
print("=" * 80)

if not os.path.exists(pdf_path):
    print(f"❌ PDF not found at: {pdf_path}")
    exit(1)

# Test 1: pdfplumber extraction
print("\n1️⃣ Testing pdfplumber extraction:")
print("-" * 80)
try:
    import pdfplumber
    
    with pdfplumber.open(pdf_path) as pdf:
        print(f"✅ PDF loaded with {len(pdf.pages)} pages")
        
        # Extract first page to see table structure
        first_page = pdf.pages[0]
        text = first_page.extract_text()
        tables = first_page.extract_tables()
        
        print(f"\n📄 Page 1 Text (first 500 chars):")
        print(text[:500] if text else "No text")
        
        print(f"\n📊 Page 1 Tables: {len(tables) if tables else 0}")
        if tables:
            for i, table in enumerate(tables[:2]):  # Show first 2 tables
                print(f"\n[TABLE {i+1}] - {len(table)} rows:")
                for row_idx, row in enumerate(table[:5]):  # Show first 5 rows
                    print(f"  Row {row_idx}: {row}")
        
        # Search for the largest budget amount
        print(f"\n🔍 Searching for budget amounts across all pages...")
        all_amounts = []
        
        for page_num, page in enumerate(pdf.pages, 1):
            page_text = page.extract_text()
            page_tables = page.extract_tables()
            
            # Look for amounts in format: ₱X,XXX,XXX.XX or PHP X,XXX,XXX.XX
            import re
            if page_text:
                # Pattern for Philippine Peso amounts
                amounts = re.findall(r'[₱PHP]\s*[\d,]+\.?\d*', page_text)
                if amounts:
                    print(f"\n  Page {page_num} amounts found: {amounts[:10]}")
                    all_amounts.extend(amounts)
            
            # Also extract from tables
            if page_tables:
                for table in page_tables:
                    for row in table:
                        if row:
                            for cell in row:
                                if cell and isinstance(cell, str):
                                    amounts = re.findall(r'[\d,]+\.?\d+', cell)
                                    for amt in amounts:
                                        if ',' in amt and len(amt) > 7:  # Likely a large amount
                                            all_amounts.append(amt)
        
        print(f"\n💰 All amounts found ({len(all_amounts)} total):")
        # Convert to numbers and find largest
        parsed_amounts = []
        for amt in all_amounts:
            try:
                # Remove currency symbols and parse
                num_str = amt.replace('₱', '').replace('PHP', '').replace(',', '').strip()
                if num_str:
                    num = float(num_str)
                    parsed_amounts.append((num, amt))
            except:
                pass
        
        # Sort by value
        parsed_amounts.sort(reverse=True)
        print("\n🏆 Top 10 largest amounts:")
        for i, (num, original) in enumerate(parsed_amounts[:10], 1):
            print(f"  {i}. {original} = {num:,.2f}")
        
        if parsed_amounts:
            largest = parsed_amounts[0]
            print(f"\n✅ LARGEST AMOUNT: {largest[1]} = ₱{largest[0]:,.2f}")
            print(f"   Expected: ₱1,084,941,000.00")
            if abs(largest[0] - 1084941000.00) < 1:
                print(f"   ✅ MATCH! Extraction is correct")
            else:
                print(f"   ❌ MISMATCH! Check extraction method")
        
except ImportError:
    print("❌ pdfplumber not installed")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: PyPDFLoader extraction
print("\n\n2️⃣ Testing PyPDFLoader extraction:")
print("-" * 80)
try:
    from langchain_community.document_loaders import PyPDFLoader
    
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    print(f"✅ Loaded {len(docs)} pages")
    print(f"\n📄 First page content (first 500 chars):")
    if docs:
        print(docs[0].page_content[:500])
        
        # Search for amounts in PyPDF output
        print(f"\n🔍 Searching for ₱1,084,941,000.00 in PyPDF extraction...")
        found = False
        for i, doc in enumerate(docs, 1):
            if "1,084,941,000" in doc.page_content or "1084941000" in doc.page_content:
                print(f"✅ FOUND on page {i}!")
                print(f"Context: {doc.page_content[max(0, doc.page_content.find('1084941000')-100):doc.page_content.find('1084941000')+100]}")
                found = True
        
        if not found:
            print("❌ Amount NOT FOUND in PyPDF extraction - tables are mangled!")
            
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 80)
print("Test complete")
print("=" * 80)
