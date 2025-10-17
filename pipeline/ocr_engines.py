"""
OCR Engines - Stage 2 of PDF Chunking Pipeline

Implements three OCR engines with async processing:
- TesseractEngine: For scanned documents
- PyPDF2Engine: For digital text extraction
- PDFPlumberEngine: For tables and structured data
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

# OCR Libraries
try:
    import pytesseract
    from PIL import Image
    from pdf2image import convert_from_path
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("Tesseract/PIL dependencies not available")

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    logging.warning("PyPDF2 not available")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logging.warning("PDFPlumber not available")

logger = logging.getLogger(__name__)

@dataclass
class OCRResult:
    """Result from OCR processing."""
    engine: str
    page_number: int
    text: str
    confidence: float
    metadata: Dict
    tables: List[Dict] = None
    bounding_boxes: List[Dict] = None
    processing_time: float = 0.0
    error: Optional[str] = None

class OCREngine(ABC):
    """Abstract base class for OCR engines."""
    
    @abstractmethod
    async def extract(self, pdf_path: Path, page_num: int) -> OCRResult:
        """Extract text from a specific page."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if engine dependencies are available."""
        pass

class TesseractEngine(OCREngine):
    """Tesseract OCR engine for scanned documents."""
    
    def __init__(self, config: str = "--oem 3 --psm 6"):
        self.config = config
        self.name = "tesseract"
    
    def is_available(self) -> bool:
        return TESSERACT_AVAILABLE
    
    async def extract(self, pdf_path: Path, page_num: int) -> OCRResult:
        """Extract text using Tesseract OCR."""
        import time
        start_time = time.time()
        
        try:
            if not self.is_available():
                return OCRResult(
                    engine=self.name,
                    page_number=page_num,
                    text="",
                    confidence=0.0,
                    metadata={},
                    error="Tesseract dependencies not available"
                )
            
            # Convert PDF page to image
            images = convert_from_path(
                pdf_path, 
                first_page=page_num + 1, 
                last_page=page_num + 1,
                dpi=300
            )
            
            if not images:
                return OCRResult(
                    engine=self.name,
                    page_number=page_num,
                    text="",
                    confidence=0.0,
                    metadata={},
                    error="Failed to convert PDF page to image"
                )
            
            image = images[0]
            
            # Preprocess image for better OCR
            preprocessed_image = self._preprocess_image(image)
            
            # Run OCR with configuration
            ocr_data = pytesseract.image_to_data(
                preprocessed_image,
                config=self.config,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text and calculate confidence
            text = pytesseract.image_to_string(preprocessed_image, config=self.config)
            confidence = self._calculate_confidence(ocr_data)
            
            # Extract bounding boxes
            bounding_boxes = self._extract_bounding_boxes(ocr_data)
            
            processing_time = time.time() - start_time
            
            return OCRResult(
                engine=self.name,
                page_number=page_num,
                text=text.strip(),
                confidence=confidence,
                metadata={
                    'image_size': image.size,
                    'dpi': 300,
                    'preprocessing': 'applied',
                    'words_detected': len([w for w in ocr_data['text'] if w.strip()])
                },
                bounding_boxes=bounding_boxes,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Tesseract OCR error on page {page_num}: {e}")
            return OCRResult(
                engine=self.name,
                page_number=page_num,
                text="",
                confidence=0.0,
                metadata={},
                processing_time=processing_time,
                error=str(e)
            )
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results."""
        try:
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Apply thresholding for better contrast
            import numpy as np
            img_array = np.array(image)
            
            # Simple thresholding
            threshold = 127
            img_array = np.where(img_array > threshold, 255, 0)
            
            # Convert back to PIL Image
            processed_image = Image.fromarray(img_array.astype(np.uint8))
            
            return processed_image
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return image
    
    def _calculate_confidence(self, ocr_data: Dict) -> float:
        """Calculate average confidence from OCR data."""
        confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
        return np.mean(confidences) / 100.0 if confidences else 0.0
    
    def _extract_bounding_boxes(self, ocr_data: Dict) -> List[Dict]:
        """Extract bounding box information."""
        boxes = []
        for i, text in enumerate(ocr_data['text']):
            if text.strip():
                boxes.append({
                    'text': text,
                    'left': ocr_data['left'][i],
                    'top': ocr_data['top'][i],
                    'width': ocr_data['width'][i],
                    'height': ocr_data['height'][i],
                    'confidence': int(ocr_data['conf'][i])
                })
        return boxes

class PyPDF2Engine(OCREngine):
    """PyPDF2 engine for digital text extraction."""
    
    def __init__(self):
        self.name = "pypdf2"
    
    def is_available(self) -> bool:
        return PYPDF2_AVAILABLE
    
    async def extract(self, pdf_path: Path, page_num: int) -> OCRResult:
        """Extract text using PyPDF2."""
        import time
        start_time = time.time()
        
        try:
            if not self.is_available():
                return OCRResult(
                    engine=self.name,
                    page_number=page_num,
                    text="",
                    confidence=0.0,
                    metadata={},
                    error="PyPDF2 not available"
                )
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                if page_num >= len(pdf_reader.pages):
                    return OCRResult(
                        engine=self.name,
                        page_number=page_num,
                        text="",
                        confidence=0.0,
                        metadata={},
                        error=f"Page {page_num} does not exist"
                    )
                
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                # Calculate confidence based on text extraction success
                confidence = self._calculate_confidence(text)
                
                processing_time = time.time() - start_time
                
                return OCRResult(
                    engine=self.name,
                    page_number=page_num,
                    text=text.strip(),
                    confidence=confidence,
                    metadata={
                        'total_pages': len(pdf_reader.pages),
                        'text_length': len(text),
                        'extraction_method': 'digital'
                    },
                    processing_time=processing_time
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"PyPDF2 error on page {page_num}: {e}")
            return OCRResult(
                engine=self.name,
                page_number=page_num,
                text="",
                confidence=0.0,
                metadata={},
                processing_time=processing_time,
                error=str(e)
            )
    
    def _calculate_confidence(self, text: str) -> float:
        """Calculate confidence based on text characteristics."""
        if not text.strip():
            return 0.0
        
        # Heuristics for digital text quality
        char_count = len(text)
        word_count = len(text.split())
        
        # Check for reasonable text characteristics
        if word_count == 0:
            return 0.0
        
        avg_word_length = char_count / word_count
        
        # Digital text typically has good structure
        confidence = 0.9
        
        # Adjust based on text characteristics
        if avg_word_length < 2:
            confidence *= 0.7  # Very short words might indicate extraction issues
        elif avg_word_length > 15:
            confidence *= 0.8  # Very long "words" might indicate poor extraction
        
        return min(confidence, 1.0)

class PDFPlumberEngine(OCREngine):
    """PDFPlumber engine for tables and structured data."""
    
    def __init__(self):
        self.name = "pdfplumber"
    
    def is_available(self) -> bool:
        return PDFPLUMBER_AVAILABLE
    
    async def extract(self, pdf_path: Path, page_num: int) -> OCRResult:
        """Extract text and tables using PDFPlumber."""
        import time
        start_time = time.time()
        
        try:
            if not self.is_available():
                return OCRResult(
                    engine=self.name,
                    page_number=page_num,
                    text="",
                    confidence=0.0,
                    metadata={},
                    error="PDFPlumber not available"
                )
            
            with pdfplumber.open(pdf_path) as pdf:
                if page_num >= len(pdf.pages):
                    return OCRResult(
                        engine=self.name,
                        page_number=page_num,
                        text="",
                        confidence=0.0,
                        metadata={},
                        error=f"Page {page_num} does not exist"
                    )
                
                page = pdf.pages[page_num]
                
                # Extract text
                text = page.extract_text() or ""
                
                # Extract tables
                tables = self._extract_tables(page)
                
                # Get bounding boxes for layout analysis
                bounding_boxes = self._extract_layout_boxes(page)
                
                confidence = self._calculate_confidence(text, tables)
                
                processing_time = time.time() - start_time
                
                return OCRResult(
                    engine=self.name,
                    page_number=page_num,
                    text=text.strip(),
                    confidence=confidence,
                    metadata={
                        'total_pages': len(pdf.pages),
                        'tables_found': len(tables),
                        'text_length': len(text),
                        'page_width': page.width,
                        'page_height': page.height
                    },
                    tables=tables,
                    bounding_boxes=bounding_boxes,
                    processing_time=processing_time
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"PDFPlumber error on page {page_num}: {e}")
            return OCRResult(
                engine=self.name,
                page_number=page_num,
                text="",
                confidence=0.0,
                metadata={},
                processing_time=processing_time,
                error=str(e)
            )
    
    def _extract_tables(self, page) -> List[Dict]:
        """Extract and structure tables from page."""
        tables = []
        
        try:
            raw_tables = page.extract_tables()
            
            for i, raw_table in enumerate(raw_tables):
                if raw_table and len(raw_table) > 1:
                    structured_table = self._structure_table(raw_table, i)
                    if structured_table:
                        tables.append(structured_table)
                        
        except Exception as e:
            logger.warning(f"Error extracting tables: {e}")
        
        return tables
    
    def _structure_table(self, raw_table: List[List[str]], table_id: int) -> Dict:
        """Convert raw table to structured format."""
        try:
            if not raw_table or len(raw_table) < 2:
                return None
            
            headers = [cell or f"col_{i}" for i, cell in enumerate(raw_table[0])]
            rows = []
            
            for row_data in raw_table[1:]:
                row_dict = {}
                for i, cell in enumerate(row_data):
                    if i < len(headers):
                        row_dict[headers[i]] = cell or ""
                
                # Only add non-empty rows
                if any(value.strip() for value in row_dict.values() if value):
                    rows.append(row_dict)
            
            return {
                'id': table_id,
                'headers': headers,
                'rows': rows,
                'row_count': len(rows),
                'col_count': len(headers),
                'table_type': 'data'
            }
            
        except Exception as e:
            logger.warning(f"Error structuring table {table_id}: {e}")
            return None
    
    def _extract_layout_boxes(self, page) -> List[Dict]:
        """Extract layout information for text positioning."""
        boxes = []
        
        try:
            # Extract character-level information for layout analysis
            chars = page.chars
            
            # Group characters into words/lines for bounding boxes
            for char in chars[:100]:  # Limit for performance
                boxes.append({
                    'text': char.get('text', ''),
                    'x0': char.get('x0', 0),
                    'y0': char.get('y0', 0),
                    'x1': char.get('x1', 0),
                    'y1': char.get('y1', 0),
                    'fontsize': char.get('size', 0),
                    'fontname': char.get('fontname', 'unknown')
                })
                
        except Exception as e:
            logger.warning(f"Error extracting layout boxes: {e}")
        
        return boxes
    
    def _calculate_confidence(self, text: str, tables: List[Dict]) -> float:
        """Calculate confidence based on extraction success."""
        confidence = 0.8  # Base confidence for PDFPlumber
        
        if text.strip():
            confidence += 0.1
        
        if tables:
            confidence += 0.1
        
        return min(confidence, 1.0)

class MultiOCRProcessor:
    """Orchestrates multiple OCR engines with async processing."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Initialize engines
        self.engines = {
            'tesseract': TesseractEngine(
                config=self.config.get('tesseract_config', "--oem 3 --psm 6")
            ),
            'pypdf2': PyPDF2Engine(),
            'pdfplumber': PDFPlumberEngine()
        }
        
        # Filter available engines
        self.available_engines = {
            name: engine for name, engine in self.engines.items()
            if engine.is_available()
        }
        
        logger.info(f"Available OCR engines: {list(self.available_engines.keys())}")
    
    async def process_page(self, pdf_path: Path, page_num: int) -> List[OCRResult]:
        """Process a single page with all available engines."""
        if not self.available_engines:
            logger.error("No OCR engines available")
            return []
        
        # Run all engines in parallel
        tasks = [
            engine.extract(pdf_path, page_num)
            for engine in self.available_engines.values()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Engine {list(self.available_engines.keys())[i]} failed: {result}")
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def process_all_pages(self, pdf_path: Path) -> Dict[int, List[OCRResult]]:
        """Process all pages of a PDF."""
        page_count = self._get_page_count(pdf_path)
        
        if page_count == 0:
            logger.error(f"Could not determine page count for {pdf_path}")
            return {}
        
        logger.info(f"Processing {page_count} pages with {len(self.available_engines)} engines")
        
        # Process pages in batches to avoid overwhelming the system
        batch_size = self.config.get('batch_size', 5)
        all_results = {}
        
        for batch_start in range(0, page_count, batch_size):
            batch_end = min(batch_start + batch_size, page_count)
            batch_tasks = []
            
            for page_num in range(batch_start, batch_end):
                task = self.process_page(pdf_path, page_num)
                batch_tasks.append((page_num, task))
            
            # Execute batch
            for page_num, task in batch_tasks:
                results = await task
                all_results[page_num] = results
                logger.info(f"Completed page {page_num + 1}/{page_count}")
        
        return all_results
    
    def compare_results(self, results: List[OCRResult]) -> OCRResult:
        """Compare results from multiple engines and return the best."""
        if not results:
            return None
        
        # Filter out results with errors
        valid_results = [r for r in results if not r.error]
        
        if not valid_results:
            # Return the first result even if it has an error
            return results[0]
        
        # Find result with highest confidence
        best_result = max(valid_results, key=lambda r: r.confidence)
        
        # Enhance with table data from PDFPlumber if available
        pdfplumber_result = next(
            (r for r in valid_results if r.engine == 'pdfplumber'), None
        )
        
        if pdfplumber_result and pdfplumber_result.tables:
            best_result.tables = pdfplumber_result.tables
        
        return best_result
    
    def merge_results(self, results: List[OCRResult]) -> OCRResult:
        """Intelligently merge results from multiple engines."""
        if not results:
            return None
        
        valid_results = [r for r in results if not r.error and r.text.strip()]
        
        if not valid_results:
            return results[0] if results else None
        
        # Use the text from the most confident result
        primary_result = max(valid_results, key=lambda r: r.confidence)
        
        # Merge metadata
        merged_metadata = {}
        for result in valid_results:
            merged_metadata.update(result.metadata)
        merged_metadata['engines_used'] = [r.engine for r in valid_results]
        merged_metadata['total_processing_time'] = sum(r.processing_time for r in valid_results)
        
        # Collect all tables
        all_tables = []
        for result in valid_results:
            if result.tables:
                all_tables.extend(result.tables)
        
        # Create merged result
        merged_result = OCRResult(
            engine='merged',
            page_number=primary_result.page_number,
            text=primary_result.text,
            confidence=primary_result.confidence,
            metadata=merged_metadata,
            tables=all_tables if all_tables else None,
            bounding_boxes=primary_result.bounding_boxes,
            processing_time=sum(r.processing_time for r in valid_results)
        )
        
        return merged_result
    
    def _get_page_count(self, pdf_path: Path) -> int:
        """Get the number of pages in a PDF."""
        try:
            if PDFPLUMBER_AVAILABLE:
                with pdfplumber.open(pdf_path) as pdf:
                    return len(pdf.pages)
            elif PYPDF2_AVAILABLE:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    return len(pdf_reader.pages)
            else:
                logger.error("No PDF libraries available to count pages")
                return 0
        except Exception as e:
            logger.error(f"Error counting pages in {pdf_path}: {e}")
            return 0