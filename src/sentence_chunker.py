"""
Stage 3: Sentence Chunking

This module implements intelligent sentence chunking using NLP sentence boundary detection.
It preserves sentence context and handles special cases like multi-line sentences in tables.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
import asyncio

# NLP libraries
import spacy
from spacy.lang.en import English

from .models import (
    RawContent, SentenceChunk, SentencePosition, 
    OCREngine, BoundingBox, DocumentMetadata
)


logger = logging.getLogger(__name__)


class SentenceChunker:
    """Splits text into sentences while preserving context and structure."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize sentence chunker with spaCy model."""
        self.config = config or {}
        
        # spaCy model for sentence boundary detection
        self.nlp = None
        self.sentencizer = None
        
        # Configuration
        self.min_sentence_length = self.config.get('min_sentence_length', 10)
        self.max_sentence_length = self.config.get('max_sentence_length', 1000)
        self.preserve_context = self.config.get('preserve_context', True)
        
        # Initialize NLP models
        self._initialize_nlp_models()
    
    def _initialize_nlp_models(self):
        """Initialize spaCy models for sentence detection."""
        try:
            # Try to load the full English model
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy en_core_web_sm model")
        except OSError:
            try:
                # Fallback to basic English sentencizer
                self.nlp = English()
                self.nlp.add_pipe("sentencizer")
                logger.info("Loaded spaCy English sentencizer")
            except Exception as e:
                logger.error(f"Failed to load spaCy models: {e}")
                # Use regex-based fallback
                self.nlp = None
        
        # Create a simple sentencizer for table text
        try:
            self.sentencizer = English()
            self.sentencizer.add_pipe("sentencizer")
        except:
            self.sentencizer = None
    
    async def chunk_into_sentences(
        self, 
        raw_content: RawContent,
        metadata: DocumentMetadata
    ) -> List[SentenceChunk]:
        """
        Split raw content into sentence chunks.
        
        Args:
            raw_content: Raw text content from OCR
            metadata: Document metadata for context
            
        Returns:
            List of SentenceChunk objects
        """
        try:
            logger.debug(f"Chunking page {raw_content.page_number} into sentences")
            
            if not raw_content.merged_text.strip():
                logger.warning(f"No text to chunk on page {raw_content.page_number}")
                return []
            
            # Split text into sentences
            sentences = await self._split_into_sentences(raw_content.merged_text)
            
            # Create sentence chunks with position information
            sentence_chunks = []
            current_position = 0
            
            for i, sentence_text in enumerate(sentences):
                if not sentence_text.strip():
                    continue
                
                # Find sentence position in original text
                start_pos = raw_content.merged_text.find(sentence_text, current_position)
                if start_pos == -1:
                    # Fallback: estimate position
                    start_pos = current_position
                
                end_pos = start_pos + len(sentence_text)
                current_position = end_pos
                
                # Create position object
                position = SentencePosition(
                    start_char=start_pos,
                    end_char=end_pos,
                    page_number=raw_content.page_number
                )
                
                # Get context (previous and next sentences)
                prev_sentence = sentences[i-1].strip() if i > 0 else None
                next_sentence = sentences[i+1].strip() if i < len(sentences)-1 else None
                
                # Create sentence chunk
                chunk = SentenceChunk(
                    text=sentence_text.strip(),
                    position=position,
                    previous_sentence=prev_sentence,
                    next_sentence=next_sentence,
                    document_id=raw_content.document_id,
                    ocr_source=raw_content.best_engine,
                    confidence=raw_content.merged_confidence
                )
                
                sentence_chunks.append(chunk)
            
            logger.debug(f"Created {len(sentence_chunks)} sentence chunks")
            return sentence_chunks
            
        except Exception as e:
            logger.error(f"Error chunking sentences: {e}")
            # Fallback: create single chunk
            if raw_content.merged_text.strip():
                fallback_chunk = SentenceChunk(
                    text=raw_content.merged_text.strip(),
                    position=SentencePosition(
                        start_char=0,
                        end_char=len(raw_content.merged_text),
                        page_number=raw_content.page_number
                    ),
                    document_id=raw_content.document_id,
                    ocr_source=raw_content.best_engine,
                    confidence=raw_content.merged_confidence
                )
                return [fallback_chunk]
            return []
    
    async def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spaCy or fallback methods."""
        if not text.strip():
            return []
        
        try:
            if self.nlp:
                return await self._spacy_sentence_split(text)
            else:
                return self._regex_sentence_split(text)
        except Exception as e:
            logger.warning(f"spaCy sentence splitting failed: {e}")
            return self._regex_sentence_split(text)
    
    async def _spacy_sentence_split(self, text: str) -> List[str]:
        """Split text into sentences using spaCy."""
        # Process text with spaCy
        doc = self.nlp(text)
        
        sentences = []
        for sent in doc.sents:
            sentence_text = sent.text.strip()
            
            # Filter by length
            if (self.min_sentence_length <= len(sentence_text) <= self.max_sentence_length):
                sentences.append(sentence_text)
            elif len(sentence_text) > self.max_sentence_length:
                # Split very long sentences
                sub_sentences = self._split_long_sentence(sentence_text)
                sentences.extend(sub_sentences)
        
        return sentences
    
    def _regex_sentence_split(self, text: str) -> List[str]:
        """Fallback sentence splitting using regex patterns."""
        # Clean text first
        text = self._clean_text_for_splitting(text)
        
        # Enhanced sentence boundary patterns
        # This pattern handles common abbreviations and edge cases
        sentence_pattern = r'''
            (?<!\w\.\w.)          # Negative lookbehind for abbreviations like "U.S.A."
            (?<![A-Z][a-z]\.)     # Negative lookbehind for names like "Mr."
            (?<![A-Z]\.)          # Negative lookbehind for single letter abbreviations
            (?<=\.|\!|\?)         # Positive lookbehind for sentence endings
            \s+                   # One or more whitespace characters
            (?=[A-Z])             # Positive lookahead for capital letter
        '''
        
        # Split using regex
        sentences = re.split(sentence_pattern, text, flags=re.VERBOSE)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if (sentence and 
                self.min_sentence_length <= len(sentence) <= self.max_sentence_length):
                cleaned_sentences.append(sentence)
            elif len(sentence) > self.max_sentence_length:
                # Split very long sentences
                sub_sentences = self._split_long_sentence(sentence)
                cleaned_sentences.extend(sub_sentences)
        
        return cleaned_sentences
    
    def _clean_text_for_splitting(self, text: str) -> str:
        """Clean text to improve sentence splitting accuracy."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR issues that affect sentence splitting
        text = re.sub(r'\.{2,}', '.', text)  # Multiple periods
        text = re.sub(r'\s+\.', '.', text)   # Space before period
        text = re.sub(r'\.\s*\n\s*([a-z])', r'. \1', text)  # Fix line breaks
        
        # Handle table-specific issues
        text = re.sub(r'\|\s*', ' ', text)   # Remove table separators
        text = re.sub(r'\s*\|\s*', ' ', text)
        
        return text.strip()
    
    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Split overly long sentences into smaller chunks."""
        if len(sentence) <= self.max_sentence_length:
            return [sentence]
        
        # Try splitting on conjunctions and punctuation
        split_patterns = [
            r',\s+(?=and\s|but\s|or\s|yet\s|so\s)',  # Conjunctions
            r';\s*',                                   # Semicolons
            r':\s*',                                   # Colons
            r',\s+(?=[A-Z])',                         # Comma before capital
            r'\s+(?=However|Moreover|Furthermore|Additionally|Therefore|Thus)',  # Transition words
        ]
        
        parts = [sentence]
        
        for pattern in split_patterns:
            new_parts = []
            for part in parts:
                if len(part) > self.max_sentence_length:
                    split_parts = re.split(pattern, part)
                    new_parts.extend([p.strip() for p in split_parts if p.strip()])
                else:
                    new_parts.append(part)
            parts = new_parts
        
        # If still too long, split by length
        final_parts = []
        for part in parts:
            if len(part) <= self.max_sentence_length:
                final_parts.append(part)
            else:
                # Split by words while respecting word boundaries
                words = part.split()
                current_chunk = []
                current_length = 0
                
                for word in words:
                    word_length = len(word) + 1  # +1 for space
                    
                    if current_length + word_length > self.max_sentence_length:
                        if current_chunk:
                            final_parts.append(' '.join(current_chunk))
                            current_chunk = [word]
                            current_length = word_length
                        else:
                            # Single word too long, truncate
                            final_parts.append(word[:self.max_sentence_length])
                    else:
                        current_chunk.append(word)
                        current_length += word_length
                
                if current_chunk:
                    final_parts.append(' '.join(current_chunk))
        
        return [part for part in final_parts if part and len(part) >= self.min_sentence_length]
    
    async def chunk_table_content(
        self, 
        table_data: List[Any], 
        raw_content: RawContent
    ) -> List[SentenceChunk]:
        """
        Handle table content chunking with special considerations.
        
        Args:
            table_data: Table data from PDFPlumber
            raw_content: Raw content context
            
        Returns:
            List of sentence chunks for table content
        """
        table_chunks = []
        
        try:
            for i, table in enumerate(table_data):
                if not hasattr(table, 'to_natural_language'):
                    continue
                
                # Convert table to natural language
                table_text = table.to_natural_language()
                
                if not table_text.strip():
                    continue
                
                # Split table description into logical parts
                table_sentences = await self._split_table_text(table_text)
                
                # Create chunks for each table part
                for j, sentence in enumerate(table_sentences):
                    if len(sentence.strip()) < self.min_sentence_length:
                        continue
                    
                    # Estimate position (tables usually appear after text)
                    estimated_start = len(raw_content.merged_text) + j * 100
                    estimated_end = estimated_start + len(sentence)
                    
                    position = SentencePosition(
                        start_char=estimated_start,
                        end_char=estimated_end,
                        page_number=raw_content.page_number,
                        bounding_box=table.bounding_box if hasattr(table, 'bounding_box') else None
                    )
                    
                    chunk = SentenceChunk(
                        text=sentence.strip(),
                        position=position,
                        document_id=raw_content.document_id,
                        ocr_source=OCREngine.PDFPLUMBER,  # Tables come from PDFPlumber
                        confidence=0.9  # Tables have high structural confidence
                    )
                    
                    table_chunks.append(chunk)
        
        except Exception as e:
            logger.error(f"Error chunking table content: {e}")
        
        return table_chunks
    
    async def _split_table_text(self, table_text: str) -> List[str]:
        """Split table natural language description into logical parts."""
        # Split on newlines first (table descriptions are often multi-line)
        lines = table_text.split('\n')
        
        sentences = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # For table content, use simpler splitting
            if self.sentencizer:
                doc = self.sentencizer(line)
                for sent in doc.sents:
                    sentence_text = sent.text.strip()
                    if len(sentence_text) >= self.min_sentence_length:
                        sentences.append(sentence_text)
            else:
                # Simple fallback for table content
                if len(line) >= self.min_sentence_length:
                    sentences.append(line)
        
        return sentences
    
    def analyze_sentence_quality(self, sentences: List[SentenceChunk]) -> Dict[str, Any]:
        """Analyze quality of sentence chunking for debugging."""
        if not sentences:
            return {
                'total_sentences': 0,
                'avg_length': 0,
                'avg_confidence': 0,
                'quality_issues': ['No sentences found']
            }
        
        lengths = [len(s.text) for s in sentences]
        confidences = [s.confidence for s in sentences]
        
        analysis = {
            'total_sentences': len(sentences),
            'avg_length': sum(lengths) / len(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'avg_confidence': sum(confidences) / len(confidences),
            'min_confidence': min(confidences),
            'sentences_with_context': sum(1 for s in sentences if s.previous_sentence or s.next_sentence),
            'quality_issues': []
        }
        
        # Identify quality issues
        short_sentences = sum(1 for l in lengths if l < 20)
        if short_sentences > len(sentences) * 0.3:
            analysis['quality_issues'].append(f"Many short sentences ({short_sentences})")
        
        long_sentences = sum(1 for l in lengths if l > 500)
        if long_sentences > 0:
            analysis['quality_issues'].append(f"Very long sentences found ({long_sentences})")
        
        low_confidence = sum(1 for c in confidences if c < 0.5)
        if low_confidence > len(sentences) * 0.2:
            analysis['quality_issues'].append(f"Low confidence sentences ({low_confidence})")
        
        if not analysis['quality_issues']:
            analysis['quality_issues'] = ['No issues detected']
        
        return analysis