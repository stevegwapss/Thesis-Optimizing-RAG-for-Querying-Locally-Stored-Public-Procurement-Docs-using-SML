"""
Advanced PDF Chunking Pipeline for RAG Systems

This package provides a comprehensive solution for processing PDF documents
into high-quality chunks suitable for Retrieval-Augmented Generation (RAG) systems.

Key Features:
- Multi-engine OCR processing (Tesseract, PyPDF2, PDFPlumber)
- Intelligent section-aware chunking
- Dual embedding generation (text and table)
- Vector database storage with rich metadata
- Async/parallel processing for performance

Main Components:
- PDFProcessor: Main orchestrator for the entire pipeline
- OCREngineManager: Manages multiple OCR engines
- SectionTagger: Identifies document structure
- EmbeddingGenerator: Creates embeddings for text and tables
- VectorStoreManager: Handles vector database operations

Example Usage:
    from src.pdf_processor import PDFProcessor
    
    processor = PDFProcessor()
    result = await processor.process_document('document.pdf')
"""

from .pdf_processor import PDFProcessor
from .models import (
    DocumentMetadata, ProcessingResult, ConfigParameters,
    ContentType, OCREngine, PDFType
)

__version__ = "1.0.0"
__author__ = "Advanced RAG System"

# Main exports
__all__ = [
    'PDFProcessor',
    'DocumentMetadata',
    'ProcessingResult', 
    'ConfigParameters',
    'ContentType',
    'OCREngine',
    'PDFType'
]