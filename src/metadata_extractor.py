"""
Stage 1: Document Metadata Extraction

This module extracts comprehensive metadata from PDF documents including:
- Basic document information (title, date, type)
- Technical properties (PDF type, page count, layout)
- Content analysis (tables, images, text distribution)
"""

import os
import re
import logging
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path

import PyPDF2
import pdfplumber
from PIL import Image
import fitz  # PyMuPDF for advanced PDF analysis

from .models import DocumentMetadata, PDFType


logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Extracts comprehensive metadata from PDF documents."""
    
    def __init__(self):
        """Initialize the metadata extractor."""
        self.title_patterns = [
            r'^([A-Z][^.]*(?:Plan|Report|Document|Procurement|Budget).*?)(?:\n|$)',
            r'^(.{1,100}?(?:FY\s*20\d{2}|Fiscal\s*Year).*?)(?:\n|$)',
            r'^([A-Z][^.]{10,80})(?:\n|$)'
        ]
        
        self.date_patterns = [
            r'FY\s*(\d{4})',
            r'Fiscal\s*Year\s*(\d{4})',
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
            r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}'
        ]
        
        self.department_patterns = [
            r'Department\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'DOH|Department\s+of\s+Health',
            r'DPWH|Department\s+of\s+Public\s+Works',
            r'DepEd|Department\s+of\s+Education',
            r'DND|Department\s+of\s+National\s+Defense'
        ]
    
    async def extract_metadata(
        self, 
        pdf_path: str, 
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentMetadata:
        """
        Extract comprehensive metadata from a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            custom_metadata: Additional metadata provided by user
            
        Returns:
            DocumentMetadata object with extracted information
        """
        try:
            logger.info(f"Extracting metadata from {pdf_path}")
            
            # Basic file information
            file_stats = os.stat(pdf_path)
            file_size = file_stats.st_size
            
            # Initialize custom fields
            custom_fields = custom_metadata or {}
            
            # Extract PDF properties and content
            pdf_properties = await self._extract_pdf_properties(pdf_path)
            content_analysis = await self._analyze_content(pdf_path)
            document_info = await self._extract_document_info(pdf_path)
            
            # Create metadata object
            metadata = DocumentMetadata(
                title=document_info.get('title'),
                date=document_info.get('date'),
                doc_type=document_info.get('doc_type'),
                department=document_info.get('department'),
                fiscal_year=document_info.get('fiscal_year'),
                
                pdf_type=pdf_properties['pdf_type'],
                page_count=pdf_properties['page_count'],
                file_size=file_size,
                file_path=str(Path(pdf_path).resolve()),
                
                has_tables=content_analysis['has_tables'],
                has_images=content_analysis['has_images'],
                estimated_text_pages=content_analysis['text_pages'],
                estimated_scanned_pages=content_analysis['scanned_pages'],
                
                custom_fields=custom_fields
            )
            
            logger.info(f"Metadata extraction completed: {metadata.page_count} pages, type: {metadata.pdf_type}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata from {pdf_path}: {str(e)}")
            raise
    
    async def _extract_pdf_properties(self, pdf_path: str) -> Dict[str, Any]:
        """Extract basic PDF properties and determine PDF type."""
        properties = {
            'page_count': 0,
            'pdf_type': PDFType.UNKNOWN,
            'creation_date': None,
            'modification_date': None,
            'producer': None,
            'creator': None
        }
        
        try:
            # Use PyPDF2 for basic properties
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                properties['page_count'] = len(pdf_reader.pages)
                
                # Extract metadata
                if pdf_reader.metadata:
                    properties['creator'] = pdf_reader.metadata.get('/Creator')
                    properties['producer'] = pdf_reader.metadata.get('/Producer')
                    
                    # Extract dates
                    creation_date = pdf_reader.metadata.get('/CreationDate')
                    if creation_date:
                        properties['creation_date'] = self._parse_pdf_date(creation_date)
            
            # Determine PDF type using multiple methods
            properties['pdf_type'] = await self._determine_pdf_type(pdf_path)
            
        except Exception as e:
            logger.warning(f"Error extracting PDF properties: {str(e)}")
            # Fallback to basic analysis
            try:
                with open(pdf_path, 'rb') as file:
                    content = file.read(10000)  # Read first 10KB
                    properties['page_count'] = content.count(b'/Page') + content.count(b'/Pages')
            except:
                properties['page_count'] = 1
        
        return properties
    
    async def _determine_pdf_type(self, pdf_path: str) -> PDFType:
        """Determine if PDF is digital, scanned, or hybrid."""
        text_ratio = 0
        image_ratio = 0
        total_pages = 0
        
        try:
            # Use PyMuPDF for detailed analysis
            pdf_doc = fitz.open(pdf_path)
            total_pages = len(pdf_doc)
            
            pages_with_text = 0
            pages_with_images = 0
            
            # Sample up to 10 pages for performance
            sample_pages = min(10, total_pages)
            step = max(1, total_pages // sample_pages)
            
            for page_num in range(0, total_pages, step):
                if page_num >= total_pages:
                    break
                    
                page = pdf_doc[page_num]
                
                # Check for text
                text = page.get_text().strip()
                if len(text) > 50:  # Minimum text threshold
                    pages_with_text += 1
                
                # Check for images
                image_list = page.get_images()
                if image_list:
                    pages_with_images += 1
            
            pdf_doc.close()
            
            # Calculate ratios
            sampled_pages = min(sample_pages, total_pages)
            text_ratio = pages_with_text / sampled_pages if sampled_pages > 0 else 0
            image_ratio = pages_with_images / sampled_pages if sampled_pages > 0 else 0
            
        except Exception as e:
            logger.warning(f"Error in PDF type detection: {str(e)}")
            # Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    total_text = ""
                    for page in pdf_reader.pages[:5]:  # Check first 5 pages
                        total_text += page.extract_text()
                    
                    text_ratio = 1.0 if len(total_text.strip()) > 100 else 0.0
            except:
                text_ratio = 0.5  # Unknown, assume hybrid
        
        # Determine PDF type based on ratios
        if text_ratio > 0.7:
            return PDFType.DIGITAL
        elif text_ratio < 0.3:
            return PDFType.SCANNED
        else:
            return PDFType.HYBRID
    
    async def _analyze_content(self, pdf_path: str) -> Dict[str, Any]:
        """Analyze content structure and estimate content types."""
        analysis = {
            'has_tables': False,
            'has_images': False,
            'text_pages': 0,
            'scanned_pages': 0,
            'table_count': 0,
            'image_count': 0
        }
        
        try:
            # Use pdfplumber for table detection
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                
                # Sample pages for performance
                sample_size = min(10, total_pages)
                sample_pages = [i * (total_pages // sample_size) for i in range(sample_size)]
                if total_pages - 1 not in sample_pages:
                    sample_pages.append(total_pages - 1)
                
                pages_with_text = 0
                total_tables = 0
                
                for page_num in sample_pages:
                    if page_num >= total_pages:
                        continue
                        
                    page = pdf.pages[page_num]
                    
                    # Check for text
                    text = page.extract_text()
                    if text and len(text.strip()) > 50:
                        pages_with_text += 1
                    
                    # Check for tables
                    tables = page.extract_tables()
                    if tables:
                        analysis['has_tables'] = True
                        total_tables += len(tables)
                
                # Estimate based on sample
                sample_ratio = len(sample_pages) / total_pages if total_pages > 0 else 1
                analysis['text_pages'] = int((pages_with_text / len(sample_pages)) * total_pages) if sample_pages else 0
                analysis['scanned_pages'] = total_pages - analysis['text_pages']
                analysis['table_count'] = int(total_tables / sample_ratio) if sample_ratio > 0 else 0
                
        except Exception as e:
            logger.warning(f"Error in content analysis: {str(e)}")
            # Basic fallback
            analysis['text_pages'] = 1
            analysis['scanned_pages'] = 0
        
        # Check for images using PyMuPDF
        try:
            pdf_doc = fitz.open(pdf_path)
            image_count = 0
            
            for page_num in range(min(5, len(pdf_doc))):  # Check first 5 pages
                page = pdf_doc[page_num]
                images = page.get_images()
                image_count += len(images)
            
            pdf_doc.close()
            
            analysis['has_images'] = image_count > 0
            analysis['image_count'] = image_count
            
        except Exception as e:
            logger.warning(f"Error checking for images: {str(e)}")
        
        return analysis
    
    async def _extract_document_info(self, pdf_path: str) -> Dict[str, Any]:
        """Extract document-specific information like title, date, department."""
        info = {
            'title': None,
            'date': None,
            'doc_type': None,
            'department': None,
            'fiscal_year': None
        }
        
        try:
            # Extract text from first few pages
            first_page_text = ""
            
            with pdfplumber.open(pdf_path) as pdf:
                # Get text from first 3 pages
                for page_num in range(min(3, len(pdf.pages))):
                    page_text = pdf.pages[page_num].extract_text()
                    if page_text:
                        first_page_text += page_text + "\n"
            
            # Clean and normalize text
            text = self._clean_text(first_page_text)
            
            # Extract title
            info['title'] = self._extract_title(text)
            
            # Extract date and fiscal year
            date_info = self._extract_date_info(text)
            info['date'] = date_info.get('date')
            info['fiscal_year'] = date_info.get('fiscal_year')
            
            # Extract department
            info['department'] = self._extract_department(text)
            
            # Determine document type
            info['doc_type'] = self._classify_document_type(text, info['title'])
            
            # Use filename as fallback for title
            if not info['title']:
                filename = Path(pdf_path).stem
                info['title'] = self._clean_filename_title(filename)
            
        except Exception as e:
            logger.warning(f"Error extracting document info: {str(e)}")
            # Use filename as fallback
            filename = Path(pdf_path).stem
            info['title'] = self._clean_filename_title(filename)
        
        return info
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and common headers/footers
        text = re.sub(r'Page\s+\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\d+\s*$', '', text)  # Remove trailing numbers
        
        return text.strip()
    
    def _extract_title(self, text: str) -> Optional[str]:
        """Extract document title using pattern matching."""
        if not text:
            return None
        
        lines = text.split('\n')
        
        # Try each title pattern
        for pattern in self.title_patterns:
            for line in lines[:10]:  # Check first 10 lines
                line = line.strip()
                if len(line) < 5:  # Skip very short lines
                    continue
                
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    title = match.group(1).strip()
                    if len(title) > 10:  # Minimum title length
                        return title
        
        # Fallback: use first substantial line
        for line in lines[:5]:
            line = line.strip()
            if len(line) > 15 and not re.match(r'^\d+$', line):
                return line
        
        return None
    
    def _extract_date_info(self, text: str) -> Dict[str, Any]:
        """Extract date and fiscal year information."""
        info = {'date': None, 'fiscal_year': None}
        
        # Extract fiscal year
        fy_match = re.search(r'FY\s*(\d{4})', text, re.IGNORECASE)
        if fy_match:
            info['fiscal_year'] = int(fy_match.group(1))
        else:
            fy_match = re.search(r'Fiscal\s*Year\s*(\d{4})', text, re.IGNORECASE)
            if fy_match:
                info['fiscal_year'] = int(fy_match.group(1))
        
        # Extract specific dates
        for pattern in self.date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    date_str = match.group(0)
                    # Try to parse the date
                    parsed_date = self._parse_date_string(date_str)
                    if parsed_date:
                        info['date'] = parsed_date
                        if not info['fiscal_year']:
                            info['fiscal_year'] = parsed_date.year
                        break
                except:
                    continue
        
        return info
    
    def _extract_department(self, text: str) -> Optional[str]:
        """Extract department information."""
        for pattern in self.department_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if len(match.groups()) > 0:
                    return match.group(1)
                else:
                    return match.group(0)
        
        return None
    
    def _classify_document_type(self, text: str, title: Optional[str]) -> Optional[str]:
        """Classify document type based on content."""
        content = (text + " " + (title or "")).lower()
        
        if any(word in content for word in ['procurement', 'plan', 'annual']):
            return 'procurement_plan'
        elif any(word in content for word in ['monitoring', 'report']):
            return 'monitoring_report'
        elif any(word in content for word in ['supplemental']):
            return 'supplemental_plan'
        elif any(word in content for word in ['budget', 'financial']):
            return 'budget_document'
        else:
            return 'document'
    
    def _parse_date_string(self, date_str: str) -> Optional[datetime]:
        """Parse various date string formats."""
        date_formats = [
            '%m/%d/%Y', '%m-%d-%Y',
            '%Y/%m/%d', '%Y-%m-%d',
            '%B %d, %Y', '%B %d %Y',
            '%d %B %Y', '%d-%b-%Y'
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def _parse_pdf_date(self, pdf_date: str) -> Optional[datetime]:
        """Parse PDF metadata date format."""
        try:
            # PDF date format: D:YYYYMMDDHHmmSSOHH'mm
            if pdf_date.startswith('D:'):
                date_part = pdf_date[2:16]  # YYYYMMDDHHMMSS
                return datetime.strptime(date_part, '%Y%m%d%H%M%S')
        except:
            pass
        
        return None
    
    def _clean_filename_title(self, filename: str) -> str:
        """Clean filename to use as title fallback."""
        # Remove file extension
        title = filename
        
        # Replace underscores and hyphens with spaces
        title = re.sub(r'[_-]', ' ', title)
        
        # Remove numbers at start
        title = re.sub(r'^\d+\.?\s*', '', title)
        
        # Capitalize properly
        title = ' '.join(word.capitalize() for word in title.split())
        
        return title