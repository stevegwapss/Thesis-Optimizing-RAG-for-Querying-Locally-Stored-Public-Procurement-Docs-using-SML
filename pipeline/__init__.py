"""
Production PDF Chunking Pipeline for RAG Systems

A sophisticated 7-stage PDF processing pipeline optimized for procurement documents:

1. File System Handling (OS + pathlib)
2. Triple OCR Extraction (Tesseract, PyPDF2, PDFPlumber)
3. spaCy Text Processing (NLP, entities, sentences)
4. spaCy PhraseMatcher (procurement-specific patterns)
5. Structured Extraction (regex + PDFPlumber tables)
6. Dual Embedding Generation (all-MiniLM-L6-v2 + TAPAS)
7. ChromaDB Vector Storage (metadata filtering, hybrid search)

Usage:
    from pipeline import PDFChunkingPipeline
    
    pipeline = PDFChunkingPipeline()
    result = await pipeline.process_document("document.pdf")
"""

from .pipeline import PDFChunkingPipeline, ProcessingResult, PipelineStats
from .file_handler import FileSystemHandler
from .ocr_engines import MultiOCRProcessor, OCRResult
from .spacy_processor import SpaCyProcessor
from .phrase_matcher import ProcurementPhraseMatcher, ContentType
from .structured_extractor import StructuredExtractor
from .embedding_generator import DualEmbedder
from .vector_store import ChromaVectorStore, Chunk, SearchResult

__version__ = "1.0.0"
__author__ = "Advanced RAG Pipeline Team"

__all__ = [
    "PDFChunkingPipeline",
    "ProcessingResult", 
    "PipelineStats",
    "FileSystemHandler",
    "MultiOCRProcessor",
    "OCRResult",
    "SpaCyProcessor",
    "ProcurementPhraseMatcher",
    "ContentType",
    "StructuredExtractor",
    "DualEmbedder",
    "ChromaVectorStore",
    "Chunk",
    "SearchResult"
]