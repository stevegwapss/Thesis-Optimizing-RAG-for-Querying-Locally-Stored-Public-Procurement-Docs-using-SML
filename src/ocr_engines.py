"""
Stage 2: OCR Engines Framework

This module implements three parallel OCR engines:
- TesseractEngine: For scanned documents using Tesseract OCR
- PyPDF2Engine: For digital PDF text extraction
- PDFPlumberEngine: For table detection and structured data extraction

Each engine provides confidence scoring and standardized output format.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import io

# OCR and PDF libraries
import pytesseract
import PyPDF2
import pdfplumber
from PIL import Image
import fitz  # PyMuPDF
import cv2
import numpy as np

from .models import (
    OCRResult, OCREngine, BoundingBox, TableData, RawContent,
    DocumentMetadata, PDFType
)


logger = logging.getLogger(__name__)


class BaseOCREngine(ABC):
    """Abstract base class for OCR engines."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize OCR engine with configuration."""
        self.config = config or {}
        self.engine_type = None
    
    @abstractmethod
    async def extract_text(
        self, 
        pdf_path: str, 
        page_number: int,
        metadata: DocumentMetadata
    ) -> OCRResult:
        """
        Extract text from a specific page.
        
        Args:
            pdf_path: Path to PDF file
            page_number: Page number (0-indexed)
            metadata: Document metadata for optimization
            
        Returns:
            OCRResult with extracted text and confidence
        """
        pass
    
    @abstractmethod
    async def extract_tables(
        self, 
        pdf_path: str, 
        page_number: int
    ) -> List[TableData]:
        """
        Extract tables from a specific page.
        
        Args:
            pdf_path: Path to PDF file
            page_number: Page number (0-indexed)
            
        Returns:
            List of TableData objects
        """
        pass
    
    def _calculate_confidence(
        self, 
        text: str, 
        context: Dict[str, Any] = None
    ) -> float:
        """Calculate confidence score for extracted text."""
        if not text:
            return 0.0
        
        context = context or {}
        confidence = 0.5  # Base confidence
        
        # Text quality indicators
        if len(text.strip()) > 10:
            confidence += 0.2
        
        # Check for common OCR errors
        error_patterns = ['���', '???', 'III', 'lll']
        error_count = sum(pattern in text for pattern in error_patterns)
        confidence -= min(0.3, error_count * 0.1)
        
        # Check text structure
        if any(char.isalpha() for char in text):
            confidence += 0.1
        
        if any(char.isdigit() for char in text):
            confidence += 0.1
        
        # Ensure confidence is within bounds
        return max(0.0, min(1.0, confidence))


class TesseractEngine(BaseOCREngine):
    """Tesseract OCR engine for scanned documents."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Tesseract engine."""
        super().__init__(config)
        self.engine_type = OCREngine.TESSERACT
        
        # Tesseract configuration
        self.tesseract_config = self.config.get(
            'tesseract_config', 
            '--oem 3 --psm 6'
        )
        
        # Check if Tesseract is available
        try:
            pytesseract.get_tesseract_version()
            self.available = True
        except Exception as e:
            logger.warning(f"Tesseract not available: {e}")
            self.available = False
    
    async def extract_text(
        self, 
        pdf_path: str, 
        page_number: int,
        metadata: DocumentMetadata
    ) -> OCRResult:
        """Extract text using Tesseract OCR."""
        start_time = time.time()
        
        try:
            if not self.available:
                return OCRResult(
                    engine=self.engine_type,
                    text="",
                    confidence=0.0,
                    processing_time=0.0,
                    success=False,
                    error_message="Tesseract not available"
                )
            
            # Convert PDF page to image
            image = await self._pdf_page_to_image(pdf_path, page_number)
            if image is None:
                raise Exception("Failed to convert PDF page to image")
            
            # Preprocess image for better OCR
            processed_image = await self._preprocess_image(image)
            
            # Extract text with confidence
            text_data = pytesseract.image_to_data(
                processed_image,
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Combine text and calculate average confidence
            text_parts = []
            confidences = []
            
            for i, word in enumerate(text_data['text']):
                if word.strip():
                    text_parts.append(word)
                    confidences.append(int(text_data['conf'][i]))
            
            extracted_text = ' '.join(text_parts)
            
            # Calculate confidence score
            if confidences:
                avg_tesseract_conf = sum(confidences) / len(confidences)
                confidence = max(0.0, min(1.0, avg_tesseract_conf / 100.0))
            else:
                confidence = 0.0
            
            # Apply additional confidence scoring
            confidence = self._calculate_confidence(
                extracted_text, 
                {'tesseract_conf': confidence}
            )
            
            processing_time = time.time() - start_time
            
            return OCRResult(
                engine=self.engine_type,
                text=extracted_text,
                confidence=confidence,
                processing_time=processing_time,
                engine_metadata={
                    'tesseract_version': pytesseract.get_tesseract_version(),
                    'config': self.tesseract_config,
                    'word_count': len(text_parts),
                    'avg_word_confidence': avg_tesseract_conf if confidences else 0
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Tesseract OCR error on page {page_number}: {str(e)}")
            
            return OCRResult(
                engine=self.engine_type,
                text="",
                confidence=0.0,
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    async def extract_tables(
        self, 
        pdf_path: str, 
        page_number: int
    ) -> List[TableData]:
        """
        Tesseract doesn't extract structured tables well.
        Returns empty list - table extraction is handled by PDFPlumber.
        """
        return []
    
    async def _pdf_page_to_image(
        self, 
        pdf_path: str, 
        page_number: int
    ) -> Optional[Image.Image]:
        """Convert PDF page to PIL Image."""
        try:
            pdf_doc = fitz.open(pdf_path)
            
            if page_number >= len(pdf_doc):
                return None
            
            page = pdf_doc[page_number]
            
            # Render page as image with high DPI for better OCR
            mat = fitz.Matrix(2.0, 2.0)  # 2x scaling for better quality
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("ppm")
            image = Image.open(io.BytesIO(img_data))
            
            pdf_doc.close()
            return image
            
        except Exception as e:
            logger.error(f"Error converting PDF page to image: {str(e)}")
            return None
    
    async def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results."""
        try:
            # Convert PIL Image to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply denoising
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Convert back to PIL Image
            processed_image = Image.fromarray(thresh)
            return processed_image
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {str(e)}")
            return image  # Return original if preprocessing fails


class PyPDF2Engine(BaseOCREngine):
    """PyPDF2 engine for digital PDF text extraction."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize PyPDF2 engine."""
        super().__init__(config)
        self.engine_type = OCREngine.PYPDF2
        self.available = True  # PyPDF2 is always available
    
    async def extract_text(
        self, 
        pdf_path: str, 
        page_number: int,
        metadata: DocumentMetadata
    ) -> OCRResult:
        """Extract text using PyPDF2."""
        start_time = time.time()
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                if page_number >= len(pdf_reader.pages):
                    raise Exception(f"Page {page_number} not found")
                
                page = pdf_reader.pages[page_number]
                extracted_text = page.extract_text()
                
                # Clean extracted text
                extracted_text = self._clean_text(extracted_text)
                
                # Calculate confidence based on text quality
                confidence = self._calculate_confidence(extracted_text)
                
                # PyPDF2 works best with digital PDFs
                if metadata.pdf_type == PDFType.DIGITAL:
                    confidence = min(1.0, confidence + 0.3)
                elif metadata.pdf_type == PDFType.SCANNED:
                    confidence = max(0.0, confidence - 0.4)
                
                processing_time = time.time() - start_time
                
                return OCRResult(
                    engine=self.engine_type,
                    text=extracted_text,
                    confidence=confidence,
                    processing_time=processing_time,
                    engine_metadata={
                        'text_length': len(extracted_text),
                        'word_count': len(extracted_text.split()),
                        'pdf_type_bonus': metadata.pdf_type == PDFType.DIGITAL
                    }
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"PyPDF2 error on page {page_number}: {str(e)}")
            
            return OCRResult(
                engine=self.engine_type,
                text="",
                confidence=0.0,
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    async def extract_tables(
        self, 
        pdf_path: str, 
        page_number: int
    ) -> List[TableData]:
        """
        PyPDF2 doesn't extract structured tables well.
        Returns empty list - table extraction is handled by PDFPlumber.
        """
        return []
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove common extraction artifacts
        text = text.replace('\x00', '')  # Null characters
        text = text.replace('\ufffd', '')  # Replacement characters
        
        return text.strip()


class PDFPlumberEngine(BaseOCREngine):
    """PDFPlumber engine for table detection and structured data extraction."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize PDFPlumber engine."""
        super().__init__(config)
        self.engine_type = OCREngine.PDFPLUMBER
        self.available = True  # pdfplumber is always available
        
        # Table detection settings
        self.min_table_rows = self.config.get('min_table_rows', 2)
        self.table_settings = self.config.get('table_settings', {
            "vertical_strategy": "lines_strict",
            "horizontal_strategy": "lines_strict",
            "intersection_tolerance": 3,
        })
    
    async def extract_text(
        self, 
        pdf_path: str, 
        page_number: int,
        metadata: DocumentMetadata
    ) -> OCRResult:
        """Extract text using PDFPlumber."""
        start_time = time.time()
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_number >= len(pdf.pages):
                    raise Exception(f"Page {page_number} not found")
                
                page = pdf.pages[page_number]
                extracted_text = page.extract_text()
                
                if not extracted_text:
                    extracted_text = ""
                
                # Calculate confidence
                confidence = self._calculate_confidence(extracted_text)
                
                # PDFPlumber generally performs well on both digital and hybrid PDFs
                if metadata.pdf_type in [PDFType.DIGITAL, PDFType.HYBRID]:
                    confidence = min(1.0, confidence + 0.2)
                
                processing_time = time.time() - start_time
                
                return OCRResult(
                    engine=self.engine_type,
                    text=extracted_text,
                    confidence=confidence,
                    processing_time=processing_time,
                    engine_metadata={
                        'text_length': len(extracted_text),
                        'word_count': len(extracted_text.split()) if extracted_text else 0,
                        'page_width': page.width,
                        'page_height': page.height
                    }
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"PDFPlumber error on page {page_number}: {str(e)}")
            
            return OCRResult(
                engine=self.engine_type,
                text="",
                confidence=0.0,
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    async def extract_tables(
        self, 
        pdf_path: str, 
        page_number: int
    ) -> List[TableData]:
        """Extract tables using PDFPlumber."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_number >= len(pdf.pages):
                    return []
                
                page = pdf.pages[page_number]
                
                # Extract tables with custom settings
                tables = page.extract_tables(self.table_settings)
                
                table_data_list = []
                
                for i, table in enumerate(tables):
                    if not table or len(table) < self.min_table_rows:
                        continue
                    
                    # First row as headers
                    headers = table[0] if table else []
                    rows = table[1:] if len(table) > 1 else []
                    
                    # Clean headers and rows
                    headers = [str(cell).strip() if cell else "" for cell in headers]
                    cleaned_rows = []
                    
                    for row in rows:
                        cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                        # Skip empty rows
                        if any(cell for cell in cleaned_row):
                            cleaned_rows.append(cleaned_row)
                    
                    if cleaned_rows:  # Only add tables with data
                        # Try to find table caption
                        caption = self._find_table_caption(page, i)
                        
                        # Create bounding box (approximate)
                        bbox = self._estimate_table_bbox(page, table)
                        
                        table_data = TableData(
                            headers=headers,
                            rows=cleaned_rows,
                            caption=caption,
                            bounding_box=bbox
                        )
                        
                        table_data_list.append(table_data)
                
                return table_data_list
                
        except Exception as e:
            logger.error(f"PDFPlumber table extraction error on page {page_number}: {str(e)}")
            return []
    
    def _find_table_caption(self, page, table_index: int) -> Optional[str]:
        """Try to find a caption for the table."""
        try:
            # Look for text above the table that might be a caption
            text = page.extract_text()
            if not text:
                return None
            
            lines = text.split('\n')
            
            # Look for lines containing "Table", "Figure", etc.
            caption_patterns = [
                r'Table\s+\d+',
                r'Figure\s+\d+',
                r'Exhibit\s+\d+',
                r'Schedule\s+\d+'
            ]
            
            import re
            for line in lines:
                for pattern in caption_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        return line.strip()
            
            return None
            
        except Exception:
            return None
    
    def _estimate_table_bbox(
        self, 
        page, 
        table: List[List[str]]
    ) -> Optional[BoundingBox]:
        """Estimate bounding box for table."""
        try:
            # This is a simplified estimation
            # In practice, you'd need more sophisticated layout analysis
            return BoundingBox(
                x0=0,
                y0=0,
                x1=page.width,
                y1=page.height / len(table) if table else page.height,
                page_number=page.page_number - 1  # Convert to 0-indexed
            )
        except Exception:
            return None


class OCREngineManager:
    """Manages multiple OCR engines and coordinates parallel execution."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize OCR engine manager."""
        self.config = config or {}
        
        # Initialize engines
        self.engines = {}
        
        # Always initialize all engines
        self.engines[OCREngine.TESSERACT] = TesseractEngine(self.config)
        self.engines[OCREngine.PYPDF2] = PyPDF2Engine(self.config)
        self.engines[OCREngine.PDFPLUMBER] = PDFPlumberEngine(self.config)
        
        # Filter available engines
        self.available_engines = {
            engine_type: engine 
            for engine_type, engine in self.engines.items() 
            if engine.available
        }
        
        logger.info(f"Initialized OCR engines: {list(self.available_engines.keys())}")
    
    async def extract_page_content(
        self, 
        pdf_path: str, 
        page_number: int,
        metadata: DocumentMetadata,
        engine_list: Optional[List[OCREngine]] = None
    ) -> RawContent:
        """
        Extract content from a page using multiple OCR engines in parallel.
        
        Args:
            pdf_path: Path to PDF file
            page_number: Page number (0-indexed)
            metadata: Document metadata
            engine_list: Specific engines to use (None for all available)
            
        Returns:
            RawContent with merged results
        """
        engines_to_use = engine_list or list(self.available_engines.keys())
        engines_to_use = [e for e in engines_to_use if e in self.available_engines]
        
        if not engines_to_use:
            raise Exception("No available OCR engines")
        
        # Run OCR engines in parallel
        ocr_tasks = []
        for engine_type in engines_to_use:
            engine = self.available_engines[engine_type]
            task = engine.extract_text(pdf_path, page_number, metadata)
            ocr_tasks.append(task)
        
        # Extract tables using PDFPlumber
        table_task = None
        if OCREngine.PDFPLUMBER in self.available_engines:
            table_task = self.available_engines[OCREngine.PDFPLUMBER].extract_tables(
                pdf_path, page_number
            )
        
        # Wait for all tasks to complete
        ocr_results = await asyncio.gather(*ocr_tasks, return_exceptions=True)
        
        # Handle OCR results
        valid_results = []
        for i, result in enumerate(ocr_results):
            if isinstance(result, Exception):
                logger.error(f"OCR engine {engines_to_use[i]} failed: {result}")
                # Create error result
                error_result = OCRResult(
                    engine=engines_to_use[i],
                    text="",
                    confidence=0.0,
                    processing_time=0.0,
                    success=False,
                    error_message=str(result)
                )
                valid_results.append(error_result)
            else:
                valid_results.append(result)
        
        # Extract tables
        tables = []
        if table_task:
            try:
                tables = await table_task
            except Exception as e:
                logger.error(f"Table extraction failed: {e}")
        
        # Create RawContent object
        raw_content = RawContent(
            document_id=metadata.document_id,
            page_number=page_number,
            ocr_results=valid_results,
            merged_text="",  # Will be filled by OCR merger
            best_engine=OCREngine.PYPDF2,  # Default, will be updated by merger
            merged_confidence=0.0,  # Will be calculated by merger
            tables=tables,
            has_text=any(result.text.strip() for result in valid_results if result.success),
            has_tables=len(tables) > 0,
            has_images=False  # Could be enhanced to detect images
        )
        
        return raw_content