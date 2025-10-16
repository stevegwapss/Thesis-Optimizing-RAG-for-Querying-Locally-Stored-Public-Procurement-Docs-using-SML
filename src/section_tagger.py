"""
Stage 4: Section Tagging

This module identifies document sections and tags sentences with their section types.
It recognizes headers, body text, tables, lists, footnotes, and maintains section hierarchy.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import uuid

from .models import (
    SentenceChunk, TaggedSentence, ContentType, TableData,
    RawContent, DocumentMetadata
)


logger = logging.getLogger(__name__)


class SectionTagger:
    """Tags sentences with their section types and maintains document structure."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize section tagger with configuration."""
        self.config = config or {}
        
        # Header detection patterns
        self.header_patterns = {
            ContentType.HEADER_H1: [
                r'^[IVX]+\.\s+[A-Z][^.]{10,80}$',  # Roman numerals
                r'^[A-Z][^.]{20,80}$',             # All caps headers
                r'^SECTION\s+[0-9]+',              # Section headers
                r'^PART\s+[IVX0-9]+',              # Part headers
                r'^CHAPTER\s+[0-9]+',              # Chapter headers
            ],
            ContentType.HEADER_H2: [
                r'^[A-Z]\.\s+[A-Z][^.]{10,60}$',   # A. Header format
                r'^[0-9]+\.\s+[A-Z][^.]{10,60}$',  # 1. Header format
                r'^[0-9]+\.[0-9]+\s+[A-Z]',       # 1.1 Header format
            ],
            ContentType.HEADER_H3: [
                r'^[a-z]\)\s+[A-Z][^.]{5,50}$',    # a) Header format
                r'^[0-9]+\.[0-9]+\.[0-9]+\s+',    # 1.1.1 Header format
                r'^\([0-9]+\)\s+[A-Z]',           # (1) Header format
            ]
        }
        
        # List patterns
        self.list_patterns = [
            r'^[•\-\*]\s+',                        # Bullet points
            r'^[0-9]+\.\s+',                       # Numbered lists
            r'^[a-z]\)\s+',                        # Lettered lists
            r'^\([a-z]\)\s+',                      # Parenthetical lists
            r'^[IVX]+\)\s+',                       # Roman numeral lists
        ]
        
        # Table context patterns
        self.table_context_patterns = [
            r'(?i)table\s+[0-9]+',
            r'(?i)exhibit\s+[a-z0-9]+',
            r'(?i)schedule\s+[a-z0-9]+',
            r'(?i)appendix\s+[a-z]+',
        ]
        
        # Footnote patterns
        self.footnote_patterns = [
            r'^\*+\s+',                            # Asterisk footnotes
            r'^[0-9]+\s+',                         # Numbered footnotes
            r'^\([0-9]+\)\s+',                     # Parenthetical footnotes
        ]
        
        # Caption patterns
        self.caption_patterns = [
            r'(?i)^(table|figure|exhibit|schedule|appendix)\s+[0-9a-z]+[:.]',
            r'(?i)^(source|note):\s+',
        ]
    
    async def tag_sentences(
        self, 
        sentences: List[SentenceChunk],
        tables: List[TableData],
        metadata: DocumentMetadata
    ) -> List[TaggedSentence]:
        """
        Tag sentences with section types and build document hierarchy.
        
        Args:
            sentences: List of sentence chunks
            tables: List of table data
            metadata: Document metadata for context
            
        Returns:
            List of TaggedSentence objects with section information
        """
        try:
            logger.debug(f"Tagging {len(sentences)} sentences with section types")
            
            if not sentences:
                return []
            
            # First pass: identify content types
            typed_sentences = await self._classify_content_types(sentences)
            
            # Second pass: build section hierarchy
            hierarchical_sentences = await self._build_section_hierarchy(typed_sentences)
            
            # Third pass: associate tables with context
            final_sentences = await self._associate_table_context(
                hierarchical_sentences, tables
            )
            
            logger.debug(f"Tagged sentences with {len(set(s.section_id for s in final_sentences))} sections")
            return final_sentences
            
        except Exception as e:
            logger.error(f"Error tagging sentences: {e}")
            # Fallback: tag everything as body text
            return [
                TaggedSentence(
                    sentence=sentence,
                    content_type=ContentType.BODY_TEXT,
                    section_id=str(uuid.uuid4()),
                    section_title="Document Content",
                    section_level=0,
                    section_order=i
                )
                for i, sentence in enumerate(sentences)
            ]
    
    async def _classify_content_types(
        self, 
        sentences: List[SentenceChunk]
    ) -> List[TaggedSentence]:
        """Classify each sentence by content type."""
        tagged_sentences = []
        
        for i, sentence in enumerate(sentences):
            content_type = await self._determine_content_type(sentence)
            
            # Create initial tagged sentence
            tagged_sentence = TaggedSentence(
                sentence=sentence,
                content_type=content_type,
                section_id="",  # Will be assigned in hierarchy building
                section_order=i
            )
            
            tagged_sentences.append(tagged_sentence)
        
        return tagged_sentences
    
    async def _determine_content_type(self, sentence: SentenceChunk) -> ContentType:
        """Determine the content type of a sentence."""
        text = sentence.text.strip()
        
        if not text:
            return ContentType.UNKNOWN
        
        # Check for headers (by pattern and position)
        for header_type, patterns in self.header_patterns.items():
            for pattern in patterns:
                if re.match(pattern, text):
                    return header_type
        
        # Check for lists
        for pattern in self.list_patterns:
            if re.match(pattern, text):
                return ContentType.LIST
        
        # Check for footnotes
        for pattern in self.footnote_patterns:
            if re.match(pattern, text) and len(text) < 200:  # Footnotes are usually short
                return ContentType.FOOTNOTE
        
        # Check for captions
        for pattern in self.caption_patterns:
            if re.match(pattern, text):
                return ContentType.CAPTION
        
        # Check for table content (heuristics)
        if self._looks_like_table_content(text):
            return ContentType.TABLE
        
        # Default to body text
        return ContentType.BODY_TEXT
    
    def _looks_like_table_content(self, text: str) -> bool:
        """Determine if text looks like it came from a table."""
        # Heuristics for table content
        indicators = [
            len(text.split()) < 10,  # Table cells are usually short
            re.search(r'\$[\d,]+', text),  # Currency amounts
            re.search(r'\d+%', text),  # Percentages
            re.search(r'\d{1,3}(,\d{3})+', text),  # Large numbers with commas
            text.count('|') > 2,  # Table separators
            re.search(r'^\d+$', text.strip()),  # Pure numbers
        ]
        
        return sum(indicators) >= 2
    
    async def _build_section_hierarchy(
        self, 
        tagged_sentences: List[TaggedSentence]
    ) -> List[TaggedSentence]:
        """Build document section hierarchy and assign section IDs."""
        current_sections = {}  # Track current section at each level
        section_counters = defaultdict(int)  # Count sections at each level
        
        for sentence in tagged_sentences:
            if sentence.content_type in [ContentType.HEADER_H1, ContentType.HEADER_H2, ContentType.HEADER_H3]:
                # This is a header - start a new section
                level = self._get_header_level(sentence.content_type)
                section_counters[level] += 1
                
                # Clear subsections when starting a new higher-level section
                levels_to_clear = [l for l in current_sections.keys() if l > level]
                for l in levels_to_clear:
                    if l in current_sections:
                        del current_sections[l]
                    section_counters[l] = 0
                
                # Create new section
                section_id = str(uuid.uuid4())
                section_title = sentence.sentence.text.strip()
                
                current_sections[level] = {
                    'id': section_id,
                    'title': section_title,
                    'level': level
                }
                
                # Update sentence with section info
                sentence.section_id = section_id
                sentence.section_title = section_title
                sentence.section_level = level
                sentence.is_section_start = True
                
                # Set parent section
                parent_level = level - 1
                if parent_level in current_sections:
                    sentence.parent_section_id = current_sections[parent_level]['id']
            
            else:
                # This is content - assign to current section
                # Find the deepest current section
                if current_sections:
                    deepest_level = max(current_sections.keys())
                    current_section = current_sections[deepest_level]
                    
                    sentence.section_id = current_section['id']
                    sentence.section_title = current_section['title']
                    sentence.section_level = current_section['level']
                    
                    # Set parent section if not at top level
                    if current_section['level'] > 0:
                        parent_level = current_section['level'] - 1
                        if parent_level in current_sections:
                            sentence.parent_section_id = current_sections[parent_level]['id']
                else:
                    # No sections defined yet - create a default section
                    default_section_id = str(uuid.uuid4())
                    sentence.section_id = default_section_id
                    sentence.section_title = "Document Content"
                    sentence.section_level = 0
        
        return tagged_sentences
    
    def _get_header_level(self, content_type: ContentType) -> int:
        """Get numeric level for header type."""
        level_map = {
            ContentType.HEADER_H1: 1,
            ContentType.HEADER_H2: 2,
            ContentType.HEADER_H3: 3,
        }
        return level_map.get(content_type, 0)
    
    async def _associate_table_context(
        self, 
        tagged_sentences: List[TaggedSentence],
        tables: List[TableData]
    ) -> List[TaggedSentence]:
        """Associate table data with relevant sentences and add context."""
        if not tables:
            return tagged_sentences
        
        # Create map of table sentences
        table_sentences = [s for s in tagged_sentences if s.content_type == ContentType.TABLE]
        
        # Try to match tables with sentences
        for i, table in enumerate(tables):
            # Find the best matching sentence for this table
            best_match = None
            best_score = 0
            
            for sentence in table_sentences:
                score = self._calculate_table_sentence_match(table, sentence)
                if score > best_score:
                    best_score = score
                    best_match = sentence
            
            # If we found a good match, associate the table
            if best_match and best_score > 0.3:
                best_match.table_data = table
                best_match.table_context = self._find_table_context(
                    best_match, tagged_sentences
                )
            else:
                # Create a new sentence for this table
                table_sentence = await self._create_table_sentence(table, tagged_sentences)
                if table_sentence:
                    tagged_sentences.append(table_sentence)
        
        return tagged_sentences
    
    def _calculate_table_sentence_match(
        self, 
        table: TableData, 
        sentence: TaggedSentence
    ) -> float:
        """Calculate how well a table matches a sentence."""
        score = 0.0
        sentence_text = sentence.sentence.text.lower()
        
        # Check if sentence mentions table-related keywords
        table_keywords = ['table', 'exhibit', 'schedule', 'data', 'amount', 'total']
        keyword_matches = sum(1 for keyword in table_keywords if keyword in sentence_text)
        score += keyword_matches * 0.2
        
        # Check if table caption matches
        if table.caption:
            caption_words = set(table.caption.lower().split())
            sentence_words = set(sentence_text.split())
            caption_overlap = len(caption_words & sentence_words) / len(caption_words) if caption_words else 0
            score += caption_overlap * 0.5
        
        # Check table content overlap
        table_text = table.to_natural_language().lower()
        table_words = set(table_text.split())
        sentence_words = set(sentence_text.split())
        content_overlap = len(table_words & sentence_words) / len(table_words) if table_words else 0
        score += content_overlap * 0.3
        
        return min(1.0, score)
    
    def _find_table_context(
        self, 
        table_sentence: TaggedSentence,
        all_sentences: List[TaggedSentence]
    ) -> Optional[str]:
        """Find contextual information around a table."""
        # Look for sentences before and after the table
        table_index = None
        for i, sentence in enumerate(all_sentences):
            if sentence.sentence.sentence_id == table_sentence.sentence.sentence_id:
                table_index = i
                break
        
        if table_index is None:
            return None
        
        context_parts = []
        
        # Look at previous 2 sentences
        for i in range(max(0, table_index - 2), table_index):
            if i < len(all_sentences):
                prev_sentence = all_sentences[i]
                if prev_sentence.content_type in [ContentType.BODY_TEXT, ContentType.CAPTION]:
                    context_parts.append(prev_sentence.sentence.text)
        
        # Look at next 2 sentences
        for i in range(table_index + 1, min(len(all_sentences), table_index + 3)):
            if i < len(all_sentences):
                next_sentence = all_sentences[i]
                if next_sentence.content_type in [ContentType.BODY_TEXT, ContentType.CAPTION]:
                    context_parts.append(next_sentence.sentence.text)
        
        return ' '.join(context_parts) if context_parts else None
    
    async def _create_table_sentence(
        self, 
        table: TableData,
        existing_sentences: List[TaggedSentence]
    ) -> Optional[TaggedSentence]:
        """Create a new sentence for a table that wasn't matched."""
        try:
            # Create a synthetic sentence chunk for the table
            table_text = table.to_natural_language()
            
            # Find appropriate section context
            section_id = "default"
            section_title = "Tables"
            section_level = 0
            
            if existing_sentences:
                # Use the last non-header sentence's section
                for sentence in reversed(existing_sentences):
                    if sentence.content_type not in [ContentType.HEADER_H1, ContentType.HEADER_H2, ContentType.HEADER_H3]:
                        section_id = sentence.section_id
                        section_title = sentence.section_title
                        section_level = sentence.section_level
                        break
            
            # Create sentence chunk
            from .models import SentencePosition, OCREngine
            
            synthetic_sentence = SentenceChunk(
                text=table_text,
                position=SentencePosition(
                    start_char=0,
                    end_char=len(table_text),
                    page_number=0,  # Unknown page
                    bounding_box=table.bounding_box
                ),
                document_id="",  # Will be set by caller
                ocr_source=OCREngine.PDFPLUMBER,
                confidence=0.9
            )
            
            # Create tagged sentence
            tagged_sentence = TaggedSentence(
                sentence=synthetic_sentence,
                content_type=ContentType.TABLE,
                section_id=section_id,
                section_title=section_title,
                section_level=section_level,
                section_order=len(existing_sentences),
                table_data=table
            )
            
            return tagged_sentence
            
        except Exception as e:
            logger.error(f"Error creating table sentence: {e}")
            return None
    
    def analyze_section_structure(
        self, 
        tagged_sentences: List[TaggedSentence]
    ) -> Dict[str, Any]:
        """Analyze the document section structure for debugging."""
        if not tagged_sentences:
            return {'error': 'No sentences to analyze'}
        
        # Count content types
        content_counts = defaultdict(int)
        for sentence in tagged_sentences:
            content_counts[sentence.content_type] += 1
        
        # Analyze sections
        sections = {}
        section_hierarchy = defaultdict(list)
        
        for sentence in tagged_sentences:
            if sentence.section_id not in sections:
                sections[sentence.section_id] = {
                    'title': sentence.section_title,
                    'level': sentence.section_level,
                    'sentence_count': 0,
                    'content_types': defaultdict(int)
                }
            
            sections[sentence.section_id]['sentence_count'] += 1
            sections[sentence.section_id]['content_types'][sentence.content_type] += 1
            section_hierarchy[sentence.section_level].append(sentence.section_id)
        
        analysis = {
            'total_sentences': len(tagged_sentences),
            'content_type_counts': dict(content_counts),
            'total_sections': len(sections),
            'section_levels': len(section_hierarchy),
            'sections': sections,
            'hierarchy': dict(section_hierarchy),
            'tables_found': sum(1 for s in tagged_sentences if s.table_data is not None),
            'sections_with_headers': sum(1 for s in sections.values() 
                                       if any(ct.name.startswith('HEADER') 
                                            for ct in s['content_types'].keys()))
        }
        
        return analysis