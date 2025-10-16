"""
Pydantic models for the advanced PDF chunking pipeline.

This module defines all data structures used throughout the RAG processing pipeline,
from initial document metadata through final vector storage.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid


class ContentType(str, Enum):
    """Types of content sections in documents."""
    HEADER_H1 = "header_h1"
    HEADER_H2 = "header_h2"
    HEADER_H3 = "header_h3"
    BODY_TEXT = "body_text"
    TABLE = "table"
    LIST = "list"
    FOOTNOTE = "footnote"
    CAPTION = "caption"
    UNKNOWN = "unknown"


class OCREngine(str, Enum):
    """Available OCR engines."""
    TESSERACT = "tesseract"
    PYPDF2 = "pypdf2"
    PDFPLUMBER = "pdfplumber"


class PDFType(str, Enum):
    """PDF document types."""
    DIGITAL = "digital"
    SCANNED = "scanned"
    HYBRID = "hybrid"


class DocumentMetadata(BaseModel):
    """Document metadata extracted in Stage 1."""
    
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: Optional[str] = None
    date: Optional[datetime] = None
    doc_type: Optional[str] = None
    department: Optional[str] = None
    fiscal_year: Optional[int] = None
    
    # Technical metadata
    pdf_type: PDFType
    page_count: int
    file_size: int  # bytes
    file_path: str
    processing_timestamp: datetime = Field(default_factory=datetime.now)
    
    # Layout information
    has_tables: bool = False
    has_images: bool = False
    estimated_text_pages: int = 0
    estimated_scanned_pages: int = 0
    
    # Custom metadata
    custom_fields: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BoundingBox(BaseModel):
    """Bounding box coordinates for text regions."""
    x0: float
    y0: float
    x1: float
    y1: float
    page_number: int


class OCRResult(BaseModel):
    """Result from a single OCR engine."""
    
    engine: OCREngine
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_box: Optional[BoundingBox] = None
    
    # Engine-specific metadata
    engine_metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_time: float  # seconds
    error_message: Optional[str] = None
    success: bool = True


class TableData(BaseModel):
    """Structured table data."""
    
    headers: List[str]
    rows: List[List[str]]
    caption: Optional[str] = None
    table_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Table position information
    bounding_box: Optional[BoundingBox] = None
    
    def to_natural_language(self) -> str:
        """Convert table to natural language description."""
        if not self.rows:
            return f"Empty table with headers: {', '.join(self.headers)}"
        
        nl_text = []
        if self.caption:
            nl_text.append(f"Table: {self.caption}")
        
        nl_text.append(f"Headers: {', '.join(self.headers)}")
        
        for i, row in enumerate(self.rows[:5]):  # Limit to first 5 rows
            row_text = ", ".join([f"{header}: {cell}" for header, cell in zip(self.headers, row)])
            nl_text.append(f"Row {i+1}: {row_text}")
        
        if len(self.rows) > 5:
            nl_text.append(f"... and {len(self.rows) - 5} more rows")
        
        return "\n".join(nl_text)


class RawContent(BaseModel):
    """Raw content extracted in Stage 2."""
    
    document_id: str
    page_number: int
    
    # OCR results from all engines
    ocr_results: List[OCRResult]
    
    # Merged content (best result from OCR engines)
    merged_text: str
    best_engine: OCREngine
    merged_confidence: float = Field(ge=0.0, le=1.0)
    
    # Extracted tables
    tables: List[TableData] = Field(default_factory=list)
    
    # Page-level metadata
    has_text: bool = True
    has_tables: bool = False
    has_images: bool = False
    
    class Config:
        arbitrary_types_allowed = True


class SentencePosition(BaseModel):
    """Position information for a sentence."""
    
    start_char: int
    end_char: int
    page_number: int
    bounding_box: Optional[BoundingBox] = None


class SentenceChunk(BaseModel):
    """Individual sentence from Stage 3."""
    
    sentence_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    position: SentencePosition
    
    # Context sentences
    previous_sentence: Optional[str] = None
    next_sentence: Optional[str] = None
    
    # Source information
    document_id: str
    ocr_source: OCREngine
    confidence: float = Field(ge=0.0, le=1.0)
    
    # Sentence properties
    word_count: int
    char_count: int
    is_complete: bool = True  # False if sentence was cut off
    
    @validator('word_count', pre=True, always=True)
    def calculate_word_count(cls, v, values):
        if 'text' in values:
            return len(values['text'].split())
        return v
    
    @validator('char_count', pre=True, always=True)
    def calculate_char_count(cls, v, values):
        if 'text' in values:
            return len(values['text'])
        return v


class TaggedSentence(BaseModel):
    """Sentence with section tagging from Stage 4."""
    
    sentence: SentenceChunk
    content_type: ContentType
    section_id: str
    section_title: Optional[str] = None
    
    # Hierarchy information
    parent_section_id: Optional[str] = None
    section_level: int = 0  # 0=root, 1=level1, etc.
    section_order: int = 0  # Order within document
    
    # Table-specific information (if content_type is TABLE)
    table_data: Optional[TableData] = None
    table_context: Optional[str] = None  # Surrounding text
    
    # Additional context
    is_section_start: bool = False
    is_section_end: bool = False


class ChunkMetadata(BaseModel):
    """Rich metadata for contextual chunks."""
    
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    page_numbers: List[int]
    
    # Position information
    start_sentence_id: str
    end_sentence_id: str
    chunk_position: int  # Order within document
    
    # Content classification
    content_types: List[ContentType]
    primary_content_type: ContentType
    is_table: bool = False
    
    # Section information
    section_ids: List[str]
    primary_section_id: str
    section_titles: List[str]
    section_hierarchy: List[int]  # Section levels
    
    # OCR source information
    ocr_sources: List[OCREngine]
    primary_ocr_source: OCREngine
    avg_confidence: float = Field(ge=0.0, le=1.0)
    
    # Size information
    token_count: int
    word_count: int
    char_count: int
    sentence_count: int
    
    # Table-specific metadata
    table_count: int = 0
    table_ids: List[str] = Field(default_factory=list)
    
    # Processing metadata
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ContextualChunk(BaseModel):
    """Section-aware chunk from Stage 5."""
    
    text: str
    metadata: ChunkMetadata
    
    # Context preservation
    overlap_previous: Optional[str] = None  # Overlapping text with previous chunk
    overlap_next: Optional[str] = None  # Overlapping text with next chunk
    
    # Structured data (for tables)
    structured_data: List[TableData] = Field(default_factory=list)
    
    # Natural language representation of tables
    table_descriptions: List[str] = Field(default_factory=list)
    
    @validator('table_descriptions', pre=True, always=True)
    def generate_table_descriptions(cls, v, values):
        if 'structured_data' in values and values['structured_data']:
            return [table.to_natural_language() for table in values['structured_data']]
        return v or []


class EmbeddingVector(BaseModel):
    """Vector embedding with metadata."""
    
    vector_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    embedding: List[float]
    model_name: str
    embedding_type: str  # "text" or "table"
    
    # Dimension validation
    dimension: int
    
    @validator('dimension', pre=True, always=True)
    def calculate_dimension(cls, v, values):
        if 'embedding' in values:
            return len(values['embedding'])
        return v


class EmbeddingResult(BaseModel):
    """Result from Stage 6 embedding generation."""
    
    chunk: ContextualChunk
    
    # Text embedding (always present)
    text_embedding: EmbeddingVector
    
    # Table embedding (only if chunk contains tables)
    table_embedding: Optional[EmbeddingVector] = None
    
    # Embedding metadata
    embedding_model: str
    embedding_timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class VectorStoreMetadata(BaseModel):
    """Metadata for vector database storage."""
    
    # Core identifiers
    vector_id: str
    document_id: str
    chunk_id: str
    
    # Content information
    text: str
    embedding_type: str  # "text" or "table"
    content_type: str
    is_table: bool
    
    # Position and structure
    page_numbers: List[int]
    chunk_position: int
    section_ids: List[str]
    section_titles: List[str]
    section_hierarchy: List[int]
    
    # Quality metrics
    ocr_source: str
    confidence_score: float
    
    # Size metrics
    token_count: int
    word_count: int
    char_count: int
    
    # Document metadata
    doc_title: Optional[str] = None
    doc_type: Optional[str] = None
    department: Optional[str] = None
    fiscal_year: Optional[int] = None
    
    # Table-specific fields
    table_count: int = 0
    table_ids: List[str] = Field(default_factory=list)
    structured_data: Optional[Dict[str, Any]] = None
    
    # Timestamps
    created_at: datetime
    indexed_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ProcessingStats(BaseModel):
    """Statistics from document processing."""
    
    total_pages: int
    total_chunks: int
    total_sentences: int
    total_tables: int
    
    # Content type breakdown
    content_type_counts: Dict[ContentType, int] = Field(default_factory=dict)
    
    # OCR engine usage
    ocr_engine_usage: Dict[OCREngine, int] = Field(default_factory=dict)
    
    # Quality metrics
    avg_confidence: float
    min_confidence: float
    max_confidence: float
    
    # Processing times
    total_processing_time: float  # seconds
    stage_times: Dict[str, float] = Field(default_factory=dict)
    
    # Error counts
    ocr_errors: int = 0
    chunking_errors: int = 0
    embedding_errors: int = 0
    
    class Config:
        use_enum_values = True


class ProcessingResult(BaseModel):
    """Final result from the complete processing pipeline."""
    
    document_metadata: DocumentMetadata
    processing_stats: ProcessingStats
    
    # Generated content
    chunks: List[ContextualChunk]
    embeddings: List[EmbeddingResult]
    
    # Vector database information
    vector_ids: List[str]
    collection_name: str
    
    # Processing metadata
    pipeline_version: str = "1.0.0"
    processing_completed_at: datetime = Field(default_factory=datetime.now)
    success: bool = True
    error_messages: List[str] = Field(default_factory=list)
    
    @property
    def chunk_count(self) -> int:
        """Total number of chunks created."""
        return len(self.chunks)
    
    @property
    def table_count(self) -> int:
        """Total number of tables detected."""
        return self.processing_stats.total_tables
    
    @property
    def avg_confidence(self) -> float:
        """Average OCR confidence score."""
        return self.processing_stats.avg_confidence
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ConfigParameters(BaseModel):
    """Configuration parameters for the processing pipeline."""
    
    # OCR settings
    ocr_engines: List[OCREngine] = Field(default=[OCREngine.TESSERACT, OCREngine.PYPDF2, OCREngine.PDFPLUMBER])
    tesseract_config: str = "--oem 3 --psm 6"
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    
    # Chunking settings
    target_chunk_size: int = Field(default=600, ge=100, le=2000)  # tokens
    max_chunk_size: int = Field(default=800, ge=200, le=3000)  # tokens
    chunk_overlap: int = Field(default=100, ge=0, le=500)  # tokens
    sentence_overlap: int = Field(default=3, ge=0, le=10)  # number of sentences
    
    # Table settings
    min_table_rows: int = Field(default=2, ge=1)
    max_table_description_length: int = Field(default=1000, ge=100)
    preserve_table_structure: bool = True
    
    # Embedding settings
    embedding_model: str = "text-embedding-3-large"
    embedding_dimension: int = 3072
    batch_size: int = Field(default=32, ge=1, le=100)
    
    # Vector database settings
    vector_store_type: str = "qdrant"  # or "weaviate"
    collection_name: str = "procurement_docs"
    distance_metric: str = "cosine"
    
    # Processing settings
    max_workers: int = Field(default=4, ge=1, le=16)
    timeout_per_page: int = Field(default=60, ge=10)  # seconds
    enable_progress_tracking: bool = True
    
    # Logging settings
    log_level: str = "INFO"
    save_intermediate_results: bool = False
    debug_mode: bool = False
    
    class Config:
        use_enum_values = True


# Type aliases for convenience
ChunkList = List[ContextualChunk]
EmbeddingList = List[EmbeddingResult]
SentenceList = List[SentenceChunk]
TaggedSentenceList = List[TaggedSentence]