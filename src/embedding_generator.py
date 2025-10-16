"""
Stage 6: Dual Embedding Generation

This module implements separate embedding strategies for text and table content.
It includes TextEmbedder for narrative content and TableEmbedder for tabular data.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Union
import time
from abc import ABC, abstractmethod

# Embedding libraries
import openai
import numpy as np

from .models import (
    ContextualChunk, EmbeddingResult, EmbeddingVector, ContentType,
    TableData, ConfigParameters
)


logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Abstract base class for embedding generators."""
    
    def __init__(self, model_name: str, config: ConfigParameters):
        """Initialize embedder with model and configuration."""
        self.model_name = model_name
        self.config = config
        self.dimension = config.embedding_dimension
        self.batch_size = config.batch_size
        
        # Initialize OpenAI client
        self.client = openai.AsyncOpenAI()
    
    @abstractmethod
    async def generate_embedding(self, text: str, context: Dict[str, Any] = None) -> List[float]:
        """Generate embedding for given text."""
        pass
    
    async def generate_batch_embeddings(
        self, 
        texts: List[str], 
        contexts: Optional[List[Dict[str, Any]]] = None
    ) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        if not texts:
            return []
        
        # Process in batches
        embeddings = []
        contexts = contexts or [{}] * len(texts)
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_contexts = contexts[i:i + self.batch_size]
            
            batch_embeddings = await self._process_batch(batch_texts, batch_contexts)
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    async def _process_batch(
        self, 
        batch_texts: List[str], 
        batch_contexts: List[Dict[str, Any]]
    ) -> List[List[float]]:
        """Process a single batch of texts."""
        tasks = []
        for text, context in zip(batch_texts, batch_contexts):
            task = self.generate_embedding(text, context)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        embeddings = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Embedding generation failed: {result}")
                # Return zero vector as fallback
                embeddings.append([0.0] * self.dimension)
            else:
                embeddings.append(result)
        
        return embeddings


class TextEmbedder(BaseEmbedder):
    """Embedding generator for narrative text content."""
    
    def __init__(self, model_name: str, config: ConfigParameters):
        """Initialize text embedder."""
        super().__init__(model_name, config)
        self.embedding_type = "text"
    
    async def generate_embedding(self, text: str, context: Dict[str, Any] = None) -> List[float]:
        """Generate embedding for text content."""
        try:
            if not text.strip():
                return [0.0] * self.dimension
            
            context = context or {}
            
            # Enhance text with context for better embeddings
            enhanced_text = self._enhance_text_with_context(text, context)
            
            # Generate embedding using OpenAI
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=enhanced_text,
                encoding_format="float"
            )
            
            embedding = response.data[0].embedding
            
            # Ensure correct dimension
            if len(embedding) != self.dimension:
                logger.warning(f"Embedding dimension mismatch: expected {self.dimension}, got {len(embedding)}")
                # Pad or truncate as needed
                if len(embedding) < self.dimension:
                    embedding.extend([0.0] * (self.dimension - len(embedding)))
                else:
                    embedding = embedding[:self.dimension]
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating text embedding: {e}")
            return [0.0] * self.dimension
    
    def _enhance_text_with_context(self, text: str, context: Dict[str, Any]) -> str:
        """Enhance text with contextual information for better embeddings."""
        enhanced_parts = []
        
        # Add section context if available
        section_title = context.get('section_title')
        if section_title:
            enhanced_parts.append(f"Section: {section_title}")
        
        # Add content type context
        content_type = context.get('content_type')
        if content_type and content_type != ContentType.BODY_TEXT:
            enhanced_parts.append(f"Content type: {content_type.value}")
        
        # Add document type context
        doc_type = context.get('doc_type')
        if doc_type:
            enhanced_parts.append(f"Document type: {doc_type}")
        
        # Add the main text
        enhanced_parts.append(text)
        
        # Add surrounding context if available
        overlap_previous = context.get('overlap_previous')
        if overlap_previous:
            enhanced_parts.insert(-1, f"Previous context: {overlap_previous}")
        
        overlap_next = context.get('overlap_next')
        if overlap_next:
            enhanced_parts.append(f"Following context: {overlap_next}")
        
        return " | ".join(enhanced_parts)


class TableEmbedder(BaseEmbedder):
    """Embedding generator for tabular data content."""
    
    def __init__(self, model_name: str, config: ConfigParameters):
        """Initialize table embedder."""
        super().__init__(model_name, config)
        self.embedding_type = "table"
        
        # Table-specific configuration
        self.max_table_description_length = config.max_table_description_length
        self.preserve_table_structure = config.preserve_table_structure
    
    async def generate_embedding(self, text: str, context: Dict[str, Any] = None) -> List[float]:
        """Generate embedding for table content."""
        try:
            if not text.strip():
                return [0.0] * self.dimension
            
            context = context or {}
            
            # Enhance table text with structure and context
            enhanced_text = self._enhance_table_with_context(text, context)
            
            # Generate embedding using OpenAI
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=enhanced_text,
                encoding_format="float"
            )
            
            embedding = response.data[0].embedding
            
            # Ensure correct dimension
            if len(embedding) != self.dimension:
                logger.warning(f"Table embedding dimension mismatch: expected {self.dimension}, got {len(embedding)}")
                if len(embedding) < self.dimension:
                    embedding.extend([0.0] * (self.dimension - len(embedding)))
                else:
                    embedding = embedding[:self.dimension]
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating table embedding: {e}")
            return [0.0] * self.dimension
    
    def _enhance_table_with_context(self, text: str, context: Dict[str, Any]) -> str:
        """Enhance table text with structural and contextual information."""
        enhanced_parts = []
        
        # Add table identifier
        enhanced_parts.append("TABLE DATA:")
        
        # Add section context
        section_title = context.get('section_title')
        if section_title:
            enhanced_parts.append(f"From section: {section_title}")
        
        # Add table structure if available
        structured_data = context.get('structured_data', [])
        if structured_data and self.preserve_table_structure:
            table_structure = self._create_structured_representation(structured_data[0])
            if table_structure:
                enhanced_parts.append(table_structure)
        
        # Add natural language description
        enhanced_parts.append(f"Description: {text}")
        
        # Add table context
        table_context = context.get('table_context')
        if table_context:
            enhanced_parts.append(f"Context: {table_context}")
        
        # Limit length
        enhanced_text = " | ".join(enhanced_parts)
        if len(enhanced_text) > self.max_table_description_length:
            enhanced_text = enhanced_text[:self.max_table_description_length]
        
        return enhanced_text
    
    def _create_structured_representation(self, table_data: TableData) -> str:
        """Create a structured text representation of table data."""
        try:
            parts = []
            
            # Add caption if available
            if table_data.caption:
                parts.append(f"Caption: {table_data.caption}")
            
            # Add headers
            if table_data.headers:
                parts.append(f"Headers: {' | '.join(table_data.headers)}")
            
            # Add sample rows (limit to avoid too much text)
            if table_data.rows:
                sample_rows = table_data.rows[:3]  # First 3 rows
                for i, row in enumerate(sample_rows):
                    row_text = " | ".join(str(cell) for cell in row)
                    parts.append(f"Row {i+1}: {row_text}")
                
                if len(table_data.rows) > 3:
                    parts.append(f"... and {len(table_data.rows) - 3} more rows")
            
            return " ".join(parts)
            
        except Exception as e:
            logger.error(f"Error creating structured representation: {e}")
            return ""


class EmbeddingGenerator:
    """Main embedding generator that coordinates text and table embedders."""
    
    def __init__(self, config: ConfigParameters):
        """Initialize embedding generator with configuration."""
        self.config = config
        
        # Initialize embedders
        self.text_embedder = TextEmbedder(config.embedding_model, config)
        self.table_embedder = TableEmbedder(config.embedding_model, config)
        
        logger.info(f"Initialized embedding generator with model: {config.embedding_model}")
    
    async def generate_embeddings(
        self, 
        chunks: List[ContextualChunk]
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for a list of contextual chunks.
        
        Args:
            chunks: List of contextual chunks to embed
            
        Returns:
            List of EmbeddingResult objects with embeddings
        """
        try:
            logger.debug(f"Generating embeddings for {len(chunks)} chunks")
            
            if not chunks:
                return []
            
            # Process chunks in batches for efficiency
            embedding_results = []
            
            for i in range(0, len(chunks), self.config.batch_size):
                batch_chunks = chunks[i:i + self.config.batch_size]
                batch_results = await self._process_chunk_batch(batch_chunks)
                embedding_results.extend(batch_results)
            
            logger.debug(f"Generated embeddings for {len(embedding_results)} chunks")
            return embedding_results
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return []
    
    async def _process_chunk_batch(
        self, 
        chunks: List[ContextualChunk]
    ) -> List[EmbeddingResult]:
        """Process a batch of chunks for embedding generation."""
        results = []
        
        # Separate text and table chunks
        text_chunks = []
        table_chunks = []
        
        for chunk in chunks:
            if chunk.metadata.is_table:
                table_chunks.append(chunk)
            else:
                text_chunks.append(chunk)
        
        # Generate text embeddings
        if text_chunks:
            text_results = await self._generate_text_embeddings(text_chunks)
            results.extend(text_results)
        
        # Generate table embeddings
        if table_chunks:
            table_results = await self._generate_table_embeddings(table_chunks)
            results.extend(table_results)
        
        return results
    
    async def _generate_text_embeddings(
        self, 
        chunks: List[ContextualChunk]
    ) -> List[EmbeddingResult]:
        """Generate embeddings for text chunks."""
        results = []
        
        # Prepare texts and contexts
        texts = []
        contexts = []
        
        for chunk in chunks:
            texts.append(chunk.text)
            
            context = {
                'section_title': chunk.metadata.section_titles[0] if chunk.metadata.section_titles else None,
                'content_type': chunk.metadata.primary_content_type,
                'doc_type': None,  # Could be added from document metadata
                'overlap_previous': chunk.overlap_previous,
                'overlap_next': chunk.overlap_next
            }
            contexts.append(context)
        
        # Generate embeddings
        embeddings = await self.text_embedder.generate_batch_embeddings(texts, contexts)
        
        # Create results
        for chunk, embedding in zip(chunks, embeddings):
            text_embedding = EmbeddingVector(
                embedding=embedding,
                model_name=self.config.embedding_model,
                embedding_type="text",
                dimension=len(embedding)
            )
            
            result = EmbeddingResult(
                chunk=chunk,
                text_embedding=text_embedding,
                embedding_model=self.config.embedding_model
            )
            
            results.append(result)
        
        return results
    
    async def _generate_table_embeddings(
        self, 
        chunks: List[ContextualChunk]
    ) -> List[EmbeddingResult]:
        """Generate embeddings for table chunks."""
        results = []
        
        # Prepare texts and contexts for table chunks
        texts = []
        contexts = []
        
        for chunk in chunks:
            # Use table descriptions if available, otherwise use main text
            if chunk.table_descriptions:
                text = " ".join(chunk.table_descriptions)
            else:
                text = chunk.text
            
            texts.append(text)
            
            context = {
                'section_title': chunk.metadata.section_titles[0] if chunk.metadata.section_titles else None,
                'structured_data': chunk.structured_data,
                'table_context': None  # Could extract from surrounding chunks
            }
            contexts.append(context)
        
        # Generate text embeddings (for natural language representation)
        text_embeddings = await self.text_embedder.generate_batch_embeddings(texts, contexts)
        
        # Generate specialized table embeddings
        table_embeddings = await self.table_embedder.generate_batch_embeddings(texts, contexts)
        
        # Create results
        for chunk, text_embedding, table_embedding in zip(chunks, text_embeddings, table_embeddings):
            text_vector = EmbeddingVector(
                embedding=text_embedding,
                model_name=self.config.embedding_model,
                embedding_type="text",
                dimension=len(text_embedding)
            )
            
            table_vector = EmbeddingVector(
                embedding=table_embedding,
                model_name=self.config.embedding_model,
                embedding_type="table",
                dimension=len(table_embedding)
            )
            
            result = EmbeddingResult(
                chunk=chunk,
                text_embedding=text_vector,
                table_embedding=table_vector,
                embedding_model=self.config.embedding_model
            )
            
            results.append(result)
        
        return results
    
    def analyze_embedding_quality(
        self, 
        embedding_results: List[EmbeddingResult]
    ) -> Dict[str, Any]:
        """Analyze the quality of generated embeddings."""
        if not embedding_results:
            return {'error': 'No embedding results to analyze'}
        
        # Basic statistics
        text_embeddings = [r.text_embedding.embedding for r in embedding_results]
        table_embeddings = [r.table_embedding.embedding for r in embedding_results if r.table_embedding]
        
        analysis = {
            'total_embeddings': len(embedding_results),
            'text_embeddings': len(text_embeddings),
            'table_embeddings': len(table_embeddings),
            'embedding_dimension': len(text_embeddings[0]) if text_embeddings else 0,
            'model_used': embedding_results[0].embedding_model if embedding_results else None
        }
        
        # Quality checks
        if text_embeddings:
            # Check for zero vectors (failed embeddings)
            zero_vectors = sum(1 for emb in text_embeddings if all(x == 0.0 for x in emb))
            analysis['zero_text_vectors'] = zero_vectors
            
            # Calculate embedding magnitudes
            magnitudes = [np.linalg.norm(emb) for emb in text_embeddings]
            analysis['magnitude_stats'] = {
                'mean': float(np.mean(magnitudes)),
                'std': float(np.std(magnitudes)),
                'min': float(np.min(magnitudes)),
                'max': float(np.max(magnitudes))
            }
        
        if table_embeddings:
            zero_table_vectors = sum(1 for emb in table_embeddings if all(x == 0.0 for x in emb))
            analysis['zero_table_vectors'] = zero_table_vectors
        
        # Recommendations
        recommendations = []
        if analysis.get('zero_text_vectors', 0) > 0:
            recommendations.append(f"{analysis['zero_text_vectors']} text embeddings failed")
        
        if analysis.get('zero_table_vectors', 0) > 0:
            recommendations.append(f"{analysis['zero_table_vectors']} table embeddings failed")
        
        if 'magnitude_stats' in analysis:
            mean_mag = analysis['magnitude_stats']['mean']
            if mean_mag < 0.1:
                recommendations.append("Embedding magnitudes are very low - check input text quality")
            elif mean_mag > 10.0:
                recommendations.append("Embedding magnitudes are very high - possible normalization issue")
        
        if not recommendations:
            recommendations = ['Embedding quality looks good']
        
        analysis['recommendations'] = recommendations
        
        return analysis