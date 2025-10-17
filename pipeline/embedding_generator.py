"""
Dual Embedding Generator - Stage 6 of PDF Chunking Pipeline

Implements two embedding strategies:
- Path A: all-MiniLM-L6-V2 for text content (lightweight, fast)
- Path B: TAPAS/T2E for tables (structure-aware)
"""

import logging
import time
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import json

# Embedding libraries
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available")

try:
    from transformers import TapasTokenizer, TapasModel, AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available")

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingResult:
    """Result from embedding generation."""
    embedding: np.ndarray
    model_name: str
    input_type: str  # 'text' or 'table'
    processing_time: float
    input_length: int
    metadata: Dict

class DualEmbedder:
    """Dual embedding generator for text and table content."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Model configurations
        self.text_model_name = self.config.get('text_model', 'all-MiniLM-L6-v2')
        self.table_model_name = self.config.get('table_model', 'google/tapas-base')
        
        # Initialize models
        self.text_model = None
        self.table_tokenizer = None
        self.table_model = None
        
        # Model metadata
        self.text_embedding_dim = 384  # all-MiniLM-L6-v2 dimension
        self.table_embedding_dim = 768  # TAPAS base dimension
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize both embedding models."""
        # Initialize text model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.text_model = SentenceTransformer(self.text_model_name)
                logger.info(f"Initialized text model: {self.text_model_name}")
            except Exception as e:
                logger.error(f"Failed to load text model {self.text_model_name}: {e}")
        else:
            logger.warning("sentence-transformers not available, text embeddings disabled")
        
        # Initialize table model
        if TRANSFORMERS_AVAILABLE:
            try:
                self.table_tokenizer = TapasTokenizer.from_pretrained(self.table_model_name)
                self.table_model = TapasModel.from_pretrained(self.table_model_name)
                logger.info(f"Initialized table model: {self.table_model_name}")
            except Exception as e:
                logger.error(f"Failed to load table model {self.table_model_name}: {e}")
                # Fallback to a simpler approach for tables
                self._initialize_fallback_table_model()
        else:
            logger.warning("transformers not available, table embeddings disabled")
    
    def _initialize_fallback_table_model(self):
        """Initialize fallback table embedding using text model."""
        if self.text_model:
            logger.info("Using text model as fallback for table embeddings")
            # We'll convert tables to text and use the text model
        else:
            logger.warning("No fallback available for table embeddings")
    
    def is_available(self) -> Dict[str, bool]:
        """Check availability of embedding models."""
        return {
            'text_embeddings': self.text_model is not None,
            'table_embeddings': (self.table_model is not None and self.table_tokenizer is not None) or self.text_model is not None
        }
    
    def embed_text(self, text: str) -> EmbeddingResult:
        """Generate embedding for text content."""
        start_time = time.time()
        
        if not self.text_model:
            return self._create_fallback_embedding(text, 'text')
        
        try:
            # Clean and validate text
            cleaned_text = self._preprocess_text(text)
            
            if not cleaned_text.strip():
                return self._create_zero_embedding('text', len(text))
            
            # Generate embedding
            embedding = self.text_model.encode(
                cleaned_text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            processing_time = time.time() - start_time
            
            return EmbeddingResult(
                embedding=embedding,
                model_name=self.text_model_name,
                input_type='text',
                processing_time=processing_time,
                input_length=len(cleaned_text),
                metadata={
                    'embedding_dim': len(embedding),
                    'normalized': True,
                    'preprocessing_applied': True
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error generating text embedding: {e}")
            return EmbeddingResult(
                embedding=np.zeros(self.text_embedding_dim),
                model_name=self.text_model_name,
                input_type='text',
                processing_time=processing_time,
                input_length=len(text),
                metadata={'error': str(e)}
            )
    
    def embed_table(self, table: Dict) -> EmbeddingResult:
        """Generate structure-aware embedding for tables."""
        start_time = time.time()
        
        # Validate table structure
        if not self._validate_table_structure(table):
            return self._create_zero_embedding('table', 0)
        
        # Try TAPAS model first
        if self.table_model and self.table_tokenizer:
            try:
                return self._embed_table_with_tapas(table, start_time)
            except Exception as e:
                logger.warning(f"TAPAS embedding failed, falling back to text approach: {e}")
        
        # Fallback to text-based table embedding
        if self.text_model:
            return self._embed_table_as_text(table, start_time)
        
        # No models available
        processing_time = time.time() - start_time
        return EmbeddingResult(
            embedding=np.zeros(self.table_embedding_dim),
            model_name='none',
            input_type='table',
            processing_time=processing_time,
            input_length=0,
            metadata={'error': 'No table embedding models available'}
        )
    
    def _embed_table_with_tapas(self, table: Dict, start_time: float) -> EmbeddingResult:
        """Embed table using TAPAS model."""
        try:
            # Convert table to TAPAS format
            table_data = self._prepare_table_for_tapas(table)
            
            # Create a query for the table (helps with contextualization)
            query = self._generate_table_query(table)
            
            # Tokenize
            inputs = self.table_tokenizer(
                table=table_data,
                queries=[query],
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.table_model(**inputs)
                # Use the CLS token embedding
                embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            
            processing_time = time.time() - start_time
            
            return EmbeddingResult(
                embedding=embedding,
                model_name=self.table_model_name,
                input_type='table',
                processing_time=processing_time,
                input_length=len(table.get('rows', [])),
                metadata={
                    'embedding_dim': len(embedding),
                    'table_rows': len(table.get('rows', [])),
                    'table_cols': len(table.get('headers', [])),
                    'method': 'tapas'
                }
            )
            
        except Exception as e:
            logger.error(f"TAPAS embedding error: {e}")
            raise
    
    def _embed_table_as_text(self, table: Dict, start_time: float) -> EmbeddingResult:
        """Embed table by converting to text and using text model."""
        try:
            # Convert table to structured text
            table_text = self._table_to_structured_text(table)
            
            # Use text model
            embedding = self.text_model.encode(
                table_text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            processing_time = time.time() - start_time
            
            return EmbeddingResult(
                embedding=embedding,
                model_name=f"{self.text_model_name}_table_fallback",
                input_type='table',
                processing_time=processing_time,
                input_length=len(table.get('rows', [])),
                metadata={
                    'embedding_dim': len(embedding),
                    'table_rows': len(table.get('rows', [])),
                    'table_cols': len(table.get('headers', [])),
                    'method': 'text_fallback',
                    'text_length': len(table_text)
                }
            )
            
        except Exception as e:
            logger.error(f"Text fallback embedding error: {e}")
            raise
    
    def batch_embed_texts(self, texts: List[str], batch_size: int = 32) -> List[EmbeddingResult]:
        """Batch embed multiple texts efficiently."""
        if not self.text_model:
            return [self._create_fallback_embedding(text, 'text') for text in texts]
        
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self._process_text_batch(batch)
            results.extend(batch_results)
        
        return results
    
    def batch_embed_tables(self, tables: List[Dict], batch_size: int = 16) -> List[EmbeddingResult]:
        """Batch embed multiple tables efficiently."""
        results = []
        
        # Process tables individually (TAPAS doesn't support batching as easily)
        for table in tables:
            result = self.embed_table(table)
            results.append(result)
        
        return results
    
    def _process_text_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Process a batch of texts."""
        start_time = time.time()
        
        try:
            # Clean texts
            cleaned_texts = [self._preprocess_text(text) for text in texts]
            
            # Generate embeddings
            embeddings = self.text_model.encode(
                cleaned_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=len(cleaned_texts)
            )
            
            processing_time = time.time() - start_time
            
            # Create results
            results = []
            for i, (original_text, cleaned_text, embedding) in enumerate(zip(texts, cleaned_texts, embeddings)):
                results.append(EmbeddingResult(
                    embedding=embedding,
                    model_name=self.text_model_name,
                    input_type='text',
                    processing_time=processing_time / len(texts),  # Distribute time
                    input_length=len(cleaned_text),
                    metadata={
                        'embedding_dim': len(embedding),
                        'batch_processed': True,
                        'batch_size': len(texts)
                    }
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Batch text embedding error: {e}")
            return [self._create_fallback_embedding(text, 'text') for text in texts]
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better embedding quality."""
        if not text:
            return ""
        
        # Basic cleaning
        cleaned = text.strip()
        
        # Remove excessive whitespace
        import re
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Truncate if too long (model limits)
        max_length = self.config.get('max_text_length', 512)
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length]
        
        return cleaned
    
    def _validate_table_structure(self, table: Dict) -> bool:
        """Validate table structure."""
        if not isinstance(table, dict):
            return False
        
        headers = table.get('headers', [])
        rows = table.get('rows', [])
        
        if not headers or not rows:
            return False
        
        # Check if rows have consistent structure
        for row in rows[:5]:  # Check first 5 rows
            if not isinstance(row, dict):
                return False
        
        return True
    
    def _prepare_table_for_tapas(self, table: Dict) -> List[List[str]]:
        """Prepare table data for TAPAS model."""
        headers = table.get('headers', [])
        rows = table.get('rows', [])
        
        # Convert to list of lists format expected by TAPAS
        table_data = [headers]
        
        for row in rows:
            row_values = []
            for header in headers:
                value = row.get(header, '')
                # Clean and convert to string
                row_values.append(str(value).strip() if value else '')
            table_data.append(row_values)
        
        return table_data
    
    def _generate_table_query(self, table: Dict) -> str:
        """Generate a query to provide context for table embedding."""
        headers = table.get('headers', [])
        metadata = table.get('metadata', {})
        
        # Create a descriptive query based on table content
        query_parts = []
        
        if metadata.get('likely_budget_table'):
            query_parts.append("budget information")
        if metadata.get('likely_timeline_table'):
            query_parts.append("timeline schedule")
        if metadata.get('table_type') == 'procurement':
            query_parts.append("procurement data")
        
        # Add header information
        if headers:
            header_desc = " ".join(headers[:3])  # First 3 headers
            query_parts.append(f"table with columns {header_desc}")
        
        # Fallback query
        if not query_parts:
            query_parts.append("data table")
        
        return " ".join(query_parts)
    
    def _table_to_structured_text(self, table: Dict) -> str:
        """Convert table to structured text representation."""
        headers = table.get('headers', [])
        rows = table.get('rows', [])
        metadata = table.get('metadata', {})
        
        text_parts = []
        
        # Add table description
        table_type = metadata.get('table_type', 'data')
        text_parts.append(f"Table: {table_type} with {len(rows)} rows and {len(headers)} columns")
        
        # Add headers
        if headers:
            text_parts.append(f"Columns: {', '.join(headers)}")
        
        # Add sample data (first few rows)
        sample_size = min(3, len(rows))
        for i, row in enumerate(rows[:sample_size]):
            row_text = []
            for header in headers:
                value = row.get(header, '')
                if value:
                    row_text.append(f"{header}: {value}")
            
            if row_text:
                text_parts.append(f"Row {i+1}: {', '.join(row_text)}")
        
        # Add summary if more rows
        if len(rows) > sample_size:
            text_parts.append(f"... and {len(rows) - sample_size} more rows")
        
        return ". ".join(text_parts)
    
    def _create_fallback_embedding(self, content: str, input_type: str) -> EmbeddingResult:
        """Create a fallback embedding when models are unavailable."""
        # Simple hash-based embedding as fallback
        import hashlib
        
        hash_obj = hashlib.md5(content.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to float array and normalize
        embedding = np.frombuffer(hash_bytes, dtype=np.uint8).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        
        # Pad or truncate to expected dimension
        expected_dim = self.text_embedding_dim if input_type == 'text' else self.table_embedding_dim
        if len(embedding) < expected_dim:
            embedding = np.pad(embedding, (0, expected_dim - len(embedding)))
        else:
            embedding = embedding[:expected_dim]
        
        return EmbeddingResult(
            embedding=embedding,
            model_name='fallback_hash',
            input_type=input_type,
            processing_time=0.001,
            input_length=len(content),
            metadata={'method': 'hash_fallback'}
        )
    
    def _create_zero_embedding(self, input_type: str, input_length: int) -> EmbeddingResult:
        """Create a zero embedding for invalid inputs."""
        dim = self.text_embedding_dim if input_type == 'text' else self.table_embedding_dim
        
        return EmbeddingResult(
            embedding=np.zeros(dim),
            model_name='zero',
            input_type=input_type,
            processing_time=0.0,
            input_length=input_length,
            metadata={'method': 'zero_embedding'}
        )
    
    def get_embedding_dimensions(self) -> Dict[str, int]:
        """Get embedding dimensions for both models."""
        return {
            'text': self.text_embedding_dim,
            'table': self.table_embedding_dim
        }
    
    def save_models_info(self, output_path: Path):
        """Save information about loaded models."""
        info = {
            'text_model': {
                'name': self.text_model_name,
                'available': self.text_model is not None,
                'dimension': self.text_embedding_dim
            },
            'table_model': {
                'name': self.table_model_name,
                'available': self.table_model is not None and self.table_tokenizer is not None,
                'dimension': self.table_embedding_dim
            },
            'libraries': {
                'sentence_transformers': SENTENCE_TRANSFORMERS_AVAILABLE,
                'transformers': TRANSFORMERS_AVAILABLE
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"Saved model info to {output_path}")
    
    def benchmark_performance(self, sample_texts: List[str], sample_tables: List[Dict]) -> Dict:
        """Benchmark embedding performance."""
        results = {
            'text_embeddings': {},
            'table_embeddings': {}
        }
        
        # Benchmark text embeddings
        if sample_texts and self.text_model:
            start_time = time.time()
            
            # Single embedding
            single_result = self.embed_text(sample_texts[0])
            single_time = single_result.processing_time
            
            # Batch embedding
            batch_results = self.batch_embed_texts(sample_texts[:5])
            batch_time = time.time() - start_time
            
            results['text_embeddings'] = {
                'single_embedding_time': single_time,
                'batch_embedding_time': batch_time,
                'avg_batch_time_per_item': batch_time / len(batch_results),
                'speedup_factor': (single_time * len(batch_results)) / batch_time if batch_time > 0 else 0
            }
        
        # Benchmark table embeddings
        if sample_tables:
            start_time = time.time()
            
            # Single table embedding
            single_result = self.embed_table(sample_tables[0])
            single_time = single_result.processing_time
            
            # Multiple table embeddings
            if len(sample_tables) > 1:
                batch_results = self.batch_embed_tables(sample_tables[:3])
                batch_time = time.time() - start_time
                
                results['table_embeddings'] = {
                    'single_embedding_time': single_time,
                    'batch_embedding_time': batch_time,
                    'avg_batch_time_per_item': batch_time / len(batch_results)
                }
        
        return results