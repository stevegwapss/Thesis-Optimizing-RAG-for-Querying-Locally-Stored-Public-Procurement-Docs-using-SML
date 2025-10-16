"""
Stage 5: Section-Aware Chunking

This module creates contextual chunks that respect document structure and section boundaries.
It implements rules for keeping sections together while maintaining optimal chunk sizes.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict

from .models import (
    TaggedSentence, ContextualChunk, ChunkMetadata, ContentType,
    TableData, ConfigParameters
)


logger = logging.getLogger(__name__)


class SectionAwareChunker:
    """Creates chunks that respect document structure and section boundaries."""
    
    def __init__(self, config: ConfigParameters):
        """Initialize chunker with configuration parameters."""
        self.config = config
        
        # Chunking parameters
        self.target_chunk_size = config.target_chunk_size
        self.max_chunk_size = config.max_chunk_size
        self.chunk_overlap = config.chunk_overlap
        self.sentence_overlap = config.sentence_overlap
        
        # Section handling rules
        self.never_split_types = {ContentType.TABLE, ContentType.CAPTION}
        self.prefer_together_types = {
            ContentType.HEADER_H1, ContentType.HEADER_H2, ContentType.HEADER_H3,
            ContentType.LIST
        }
        
        # Token estimation (rough approximation)
        self.tokens_per_char = 0.25  # Rough estimate for English text
    
    async def create_contextual_chunks(
        self, 
        tagged_sentences: List[TaggedSentence]
    ) -> List[ContextualChunk]:
        """
        Create contextual chunks from tagged sentences.
        
        Args:
            tagged_sentences: List of sentences with section tags
            
        Returns:
            List of ContextualChunk objects
        """
        try:
            logger.debug(f"Creating contextual chunks from {len(tagged_sentences)} sentences")
            
            if not tagged_sentences:
                return []
            
            # Group sentences by section for processing
            section_groups = self._group_by_section(tagged_sentences)
            
            # Process each section group
            all_chunks = []
            chunk_position = 0
            
            for section_id, sentences in section_groups.items():
                section_chunks = await self._process_section_group(
                    sentences, chunk_position
                )
                all_chunks.extend(section_chunks)
                chunk_position += len(section_chunks)
            
            # Add overlap between chunks
            overlapped_chunks = await self._add_chunk_overlap(all_chunks)
            
            logger.debug(f"Created {len(overlapped_chunks)} contextual chunks")
            return overlapped_chunks
            
        except Exception as e:
            logger.error(f"Error creating contextual chunks: {e}")
            # Fallback: create simple chunks
            return await self._create_fallback_chunks(tagged_sentences)
    
    def _group_by_section(
        self, 
        tagged_sentences: List[TaggedSentence]
    ) -> Dict[str, List[TaggedSentence]]:
        """Group sentences by their section IDs."""
        section_groups = defaultdict(list)
        
        for sentence in tagged_sentences:
            section_groups[sentence.section_id].append(sentence)
        
        # Sort sentences within each section by order
        for section_id in section_groups:
            section_groups[section_id].sort(key=lambda s: s.section_order)
        
        return dict(section_groups)
    
    async def _process_section_group(
        self, 
        sentences: List[TaggedSentence],
        start_chunk_position: int
    ) -> List[ContextualChunk]:
        """Process a group of sentences from the same section."""
        if not sentences:
            return []
        
        # Separate special content types
        tables = [s for s in sentences if s.content_type == ContentType.TABLE]
        headers = [s for s in sentences if s.content_type in self.prefer_together_types]
        regular_content = [s for s in sentences if s not in tables and s not in headers]
        
        chunks = []
        chunk_position = start_chunk_position
        
        # Process tables first (they get their own chunks)
        for table_sentence in tables:
            table_chunk = await self._create_table_chunk(table_sentence, chunk_position)
            if table_chunk:
                chunks.append(table_chunk)
                chunk_position += 1
        
        # Process headers with their following content
        if headers or regular_content:
            content_chunks = await self._create_content_chunks(
                headers + regular_content, chunk_position
            )
            chunks.extend(content_chunks)
        
        return chunks
    
    async def _create_table_chunk(
        self, 
        table_sentence: TaggedSentence,
        chunk_position: int
    ) -> Optional[ContextualChunk]:
        """Create a dedicated chunk for table content."""
        try:
            # Tables always get their own chunk
            text = table_sentence.sentence.text
            
            # Collect table data if available
            structured_data = []
            if table_sentence.table_data:
                structured_data.append(table_sentence.table_data)
            
            # Create metadata
            metadata = ChunkMetadata(
                document_id=table_sentence.sentence.document_id,
                page_numbers=[table_sentence.sentence.position.page_number],
                start_sentence_id=table_sentence.sentence.sentence_id,
                end_sentence_id=table_sentence.sentence.sentence_id,
                chunk_position=chunk_position,
                content_types=[ContentType.TABLE],
                primary_content_type=ContentType.TABLE,
                is_table=True,
                section_ids=[table_sentence.section_id],
                primary_section_id=table_sentence.section_id,
                section_titles=[table_sentence.section_title or ""],
                section_hierarchy=[table_sentence.section_level],
                ocr_sources=[table_sentence.sentence.ocr_source],
                primary_ocr_source=table_sentence.sentence.ocr_source,
                avg_confidence=table_sentence.sentence.confidence,
                token_count=self._estimate_tokens(text),
                word_count=len(text.split()),
                char_count=len(text),
                sentence_count=1,
                table_count=1 if table_sentence.table_data else 0,
                table_ids=[table_sentence.table_data.table_id] if table_sentence.table_data else []
            )
            
            # Create chunk
            chunk = ContextualChunk(
                text=text,
                metadata=metadata,
                structured_data=structured_data
            )
            
            return chunk
            
        except Exception as e:
            logger.error(f"Error creating table chunk: {e}")
            return None
    
    async def _create_content_chunks(
        self, 
        sentences: List[TaggedSentence],
        start_chunk_position: int
    ) -> List[ContextualChunk]:
        """Create chunks for regular content, respecting section boundaries."""
        if not sentences:
            return []
        
        chunks = []
        current_chunk_sentences = []
        current_chunk_size = 0
        chunk_position = start_chunk_position
        
        for i, sentence in enumerate(sentences):
            sentence_size = self._estimate_tokens(sentence.sentence.text)
            
            # Check if adding this sentence would exceed chunk size
            if (current_chunk_size + sentence_size > self.target_chunk_size and 
                current_chunk_sentences):
                
                # Create chunk from current sentences
                chunk = await self._create_chunk_from_sentences(
                    current_chunk_sentences, chunk_position
                )
                if chunk:
                    chunks.append(chunk)
                    chunk_position += 1
                
                # Start new chunk
                current_chunk_sentences = [sentence]
                current_chunk_size = sentence_size
            
            else:
                # Add sentence to current chunk
                current_chunk_sentences.append(sentence)
                current_chunk_size += sentence_size
            
            # Force chunk creation if we exceed max size
            if current_chunk_size > self.max_chunk_size:
                chunk = await self._create_chunk_from_sentences(
                    current_chunk_sentences, chunk_position
                )
                if chunk:
                    chunks.append(chunk)
                    chunk_position += 1
                
                current_chunk_sentences = []
                current_chunk_size = 0
        
        # Create final chunk if there are remaining sentences
        if current_chunk_sentences:
            chunk = await self._create_chunk_from_sentences(
                current_chunk_sentences, chunk_position
            )
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    async def _create_chunk_from_sentences(
        self, 
        sentences: List[TaggedSentence],
        chunk_position: int
    ) -> Optional[ContextualChunk]:
        """Create a chunk from a list of sentences."""
        try:
            if not sentences:
                return None
            
            # Combine sentence texts
            text_parts = [s.sentence.text for s in sentences]
            combined_text = ' '.join(text_parts)
            
            # Collect metadata
            page_numbers = list(set(s.sentence.position.page_number for s in sentences))
            content_types = list(set(s.content_type for s in sentences))
            section_ids = list(set(s.section_id for s in sentences))
            section_titles = list(set(s.section_title for s in sentences if s.section_title))
            section_hierarchy = list(set(s.section_level for s in sentences))
            ocr_sources = list(set(s.sentence.ocr_source for s in sentences))
            
            # Determine primary values
            primary_content_type = self._determine_primary_content_type(content_types)
            primary_section_id = section_ids[0] if section_ids else ""
            primary_ocr_source = ocr_sources[0] if ocr_sources else sentences[0].sentence.ocr_source
            
            # Calculate confidence
            confidences = [s.sentence.confidence for s in sentences]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Collect table data
            structured_data = []
            table_ids = []
            table_count = 0
            
            for sentence in sentences:
                if sentence.table_data:
                    structured_data.append(sentence.table_data)
                    table_ids.append(sentence.table_data.table_id)
                    table_count += 1
            
            # Create metadata
            metadata = ChunkMetadata(
                document_id=sentences[0].sentence.document_id,
                page_numbers=sorted(page_numbers),
                start_sentence_id=sentences[0].sentence.sentence_id,
                end_sentence_id=sentences[-1].sentence.sentence_id,
                chunk_position=chunk_position,
                content_types=content_types,
                primary_content_type=primary_content_type,
                is_table=ContentType.TABLE in content_types,
                section_ids=section_ids,
                primary_section_id=primary_section_id,
                section_titles=section_titles,
                section_hierarchy=sorted(section_hierarchy),
                ocr_sources=ocr_sources,
                primary_ocr_source=primary_ocr_source,
                avg_confidence=avg_confidence,
                token_count=self._estimate_tokens(combined_text),
                word_count=len(combined_text.split()),
                char_count=len(combined_text),
                sentence_count=len(sentences),
                table_count=table_count,
                table_ids=table_ids
            )
            
            # Create chunk
            chunk = ContextualChunk(
                text=combined_text,
                metadata=metadata,
                structured_data=structured_data
            )
            
            return chunk
            
        except Exception as e:
            logger.error(f"Error creating chunk from sentences: {e}")
            return None
    
    def _determine_primary_content_type(
        self, 
        content_types: List[ContentType]
    ) -> ContentType:
        """Determine the primary content type for a chunk."""
        # Priority order
        priority = [
            ContentType.TABLE,
            ContentType.HEADER_H1,
            ContentType.HEADER_H2,
            ContentType.HEADER_H3,
            ContentType.LIST,
            ContentType.CAPTION,
            ContentType.BODY_TEXT,
            ContentType.FOOTNOTE,
            ContentType.UNKNOWN
        ]
        
        for content_type in priority:
            if content_type in content_types:
                return content_type
        
        return ContentType.BODY_TEXT
    
    async def _add_chunk_overlap(
        self, 
        chunks: List[ContextualChunk]
    ) -> List[ContextualChunk]:
        """Add overlap between consecutive chunks."""
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Get overlap from previous chunk
            overlap_previous = None
            if i > 0:
                prev_chunk = chunks[i-1]
                overlap_previous = self._extract_overlap_text(
                    prev_chunk.text, from_end=True
                )
            
            # Get overlap from next chunk
            overlap_next = None
            if i < len(chunks) - 1:
                next_chunk = chunks[i+1]
                overlap_next = self._extract_overlap_text(
                    next_chunk.text, from_end=False
                )
            
            # Create new chunk with overlap
            new_chunk = ContextualChunk(
                text=chunk.text,
                metadata=chunk.metadata,
                overlap_previous=overlap_previous,
                overlap_next=overlap_next,
                structured_data=chunk.structured_data,
                table_descriptions=chunk.table_descriptions
            )
            
            overlapped_chunks.append(new_chunk)
        
        return overlapped_chunks
    
    def _extract_overlap_text(self, text: str, from_end: bool) -> Optional[str]:
        """Extract overlap text from beginning or end of text."""
        if not text or self.chunk_overlap <= 0:
            return None
        
        # Split into sentences for better overlap
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if from_end:
            # Take last few sentences
            overlap_sentences = sentences[-self.sentence_overlap:] if len(sentences) > self.sentence_overlap else sentences
        else:
            # Take first few sentences
            overlap_sentences = sentences[:self.sentence_overlap] if len(sentences) > self.sentence_overlap else sentences
        
        overlap_text = ' '.join(overlap_sentences)
        
        # Limit by character count
        if len(overlap_text) > self.chunk_overlap:
            if from_end:
                overlap_text = overlap_text[-self.chunk_overlap:]
            else:
                overlap_text = overlap_text[:self.chunk_overlap]
        
        return overlap_text if overlap_text.strip() else None
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        if not text:
            return 0
        
        # Simple estimation: roughly 4 characters per token
        return int(len(text) * self.tokens_per_char)
    
    async def _create_fallback_chunks(
        self, 
        tagged_sentences: List[TaggedSentence]
    ) -> List[ContextualChunk]:
        """Create simple chunks as fallback when section-aware chunking fails."""
        try:
            chunks = []
            current_sentences = []
            current_size = 0
            
            for i, sentence in enumerate(tagged_sentences):
                sentence_size = self._estimate_tokens(sentence.sentence.text)
                
                if current_size + sentence_size > self.target_chunk_size and current_sentences:
                    # Create chunk
                    chunk = await self._create_chunk_from_sentences(current_sentences, len(chunks))
                    if chunk:
                        chunks.append(chunk)
                    
                    current_sentences = [sentence]
                    current_size = sentence_size
                else:
                    current_sentences.append(sentence)
                    current_size += sentence_size
            
            # Final chunk
            if current_sentences:
                chunk = await self._create_chunk_from_sentences(current_sentences, len(chunks))
                if chunk:
                    chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating fallback chunks: {e}")
            return []
    
    def analyze_chunking_quality(
        self, 
        chunks: List[ContextualChunk]
    ) -> Dict[str, Any]:
        """Analyze the quality of chunking for debugging and optimization."""
        if not chunks:
            return {'error': 'No chunks to analyze'}
        
        # Size analysis
        token_counts = [chunk.metadata.token_count for chunk in chunks]
        char_counts = [chunk.metadata.char_count for chunk in chunks]
        sentence_counts = [chunk.metadata.sentence_count for chunk in chunks]
        
        # Content type analysis
        content_type_dist = defaultdict(int)
        section_dist = defaultdict(int)
        
        for chunk in chunks:
            content_type_dist[chunk.metadata.primary_content_type] += 1
            for section_id in chunk.metadata.section_ids:
                section_dist[section_id] += 1
        
        # Quality metrics
        optimal_size_chunks = sum(1 for count in token_counts 
                                if self.target_chunk_size * 0.8 <= count <= self.target_chunk_size * 1.2)
        oversized_chunks = sum(1 for count in token_counts if count > self.max_chunk_size)
        
        analysis = {
            'total_chunks': len(chunks),
            'size_stats': {
                'avg_tokens': sum(token_counts) / len(token_counts),
                'min_tokens': min(token_counts),
                'max_tokens': max(token_counts),
                'avg_chars': sum(char_counts) / len(char_counts),
                'avg_sentences': sum(sentence_counts) / len(sentence_counts)
            },
            'content_distribution': dict(content_type_dist),
            'section_distribution': len(section_dist),
            'quality_metrics': {
                'optimal_size_ratio': optimal_size_chunks / len(chunks),
                'oversized_chunks': oversized_chunks,
                'chunks_with_tables': sum(1 for chunk in chunks if chunk.metadata.is_table),
                'chunks_with_overlap': sum(1 for chunk in chunks 
                                         if chunk.overlap_previous or chunk.overlap_next),
                'multi_section_chunks': sum(1 for chunk in chunks 
                                          if len(chunk.metadata.section_ids) > 1)
            },
            'recommendations': []
        }
        
        # Generate recommendations
        if analysis['quality_metrics']['optimal_size_ratio'] < 0.5:
            analysis['recommendations'].append("Consider adjusting target chunk size")
        
        if oversized_chunks > 0:
            analysis['recommendations'].append(f"{oversized_chunks} chunks exceed max size")
        
        if analysis['quality_metrics']['chunks_with_overlap'] / len(chunks) < 0.5:
            analysis['recommendations'].append("Consider increasing chunk overlap")
        
        if not analysis['recommendations']:
            analysis['recommendations'] = ['Chunking quality looks good']
        
        return analysis