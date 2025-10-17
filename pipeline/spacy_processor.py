"""
spaCy Text Processor - Stage 3 of PDF Chunking Pipeline

Uses spaCy for intelligent text processing:
- Merge OCR results intelligently
- Sentence segmentation
- Token analysis
- Named entity recognition
- Dependency parsing
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import re

try:
    import spacy
    from spacy.lang.en import English
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available")

logger = logging.getLogger(__name__)

@dataclass
class Sentence:
    """Represents a processed sentence."""
    text: str
    start_char: int
    end_char: int
    tokens: List[str]
    entities: List[Dict]
    metadata: Dict

@dataclass
class Entity:
    """Represents a named entity."""
    text: str
    label: str
    start_char: int
    end_char: int
    confidence: float

@dataclass
class DocumentStructure:
    """Represents the analyzed structure of a document."""
    sentences: List[Sentence]
    entities: List[Entity]
    sections: List[Dict]
    metadata: Dict

class SpaCyProcessor:
    """Intelligent text processing using spaCy."""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        self.nlp = None
        self._initialize_nlp()
    
    def _initialize_nlp(self):
        """Initialize spaCy pipeline."""
        if not SPACY_AVAILABLE:
            logger.error("spaCy not available")
            return
        
        try:
            # Load the spaCy model
            self.nlp = spacy.load(self.model_name)
            
            # Add custom components for our use case
            if 'sentencizer' not in self.nlp.pipe_names:
                self.nlp.add_pipe('sentencizer')
            
            # Configure pipeline for efficiency
            self.nlp.max_length = 2000000  # Handle large documents
            
            logger.info(f"Initialized spaCy with model: {self.model_name}")
            logger.info(f"Pipeline components: {self.nlp.pipe_names}")
            
        except OSError:
            logger.error(f"spaCy model '{self.model_name}' not found. Try: python -m spacy download {self.model_name}")
            # Fallback to basic English
            try:
                self.nlp = English()
                self.nlp.add_pipe('sentencizer')
                logger.warning("Using basic English pipeline as fallback")
            except Exception as e:
                logger.error(f"Failed to initialize any spaCy pipeline: {e}")
    
    def is_available(self) -> bool:
        """Check if spaCy processor is available."""
        return SPACY_AVAILABLE and self.nlp is not None
    
    def merge_ocr_results(self, ocr_results: List[Dict]) -> str:
        """Intelligently merge OCR results from multiple engines."""
        if not ocr_results:
            return ""
        
        # Filter valid results
        valid_results = [r for r in ocr_results if r.get('text', '').strip() and not r.get('error')]
        
        if not valid_results:
            # Return first result even if it has issues
            return ocr_results[0].get('text', '')
        
        # Strategy 1: Use highest confidence result
        best_result = max(valid_results, key=lambda r: r.get('confidence', 0.0))
        primary_text = best_result.get('text', '')
        
        # Strategy 2: Merge table data from PDFPlumber
        table_data = []
        for result in valid_results:
            if result.get('engine') == 'pdfplumber' and result.get('tables'):
                table_data.extend(result['tables'])
        
        # Combine text with table data
        if table_data:
            table_text = self._tables_to_text(table_data)
            if table_text:
                primary_text = f"{primary_text}\n\n{table_text}"
        
        # Strategy 3: Fill gaps using other engines
        merged_text = self._fill_text_gaps(primary_text, valid_results)
        
        return merged_text.strip()
    
    def _tables_to_text(self, tables: List[Dict]) -> str:
        """Convert table data to readable text."""
        table_texts = []
        
        for table in tables:
            if not table.get('rows'):
                continue
            
            headers = table.get('headers', [])
            rows = table.get('rows', [])
            
            # Create table text representation
            table_text = f"Table {table.get('id', 'Unknown')}:\n"
            
            if headers:
                table_text += "Headers: " + " | ".join(headers) + "\n"
            
            for i, row in enumerate(rows[:5]):  # Limit to first 5 rows
                row_text = " | ".join([str(row.get(header, '')) for header in headers])
                table_text += f"Row {i+1}: {row_text}\n"
            
            if len(rows) > 5:
                table_text += f"... and {len(rows) - 5} more rows\n"
            
            table_texts.append(table_text)
        
        return "\n".join(table_texts)
    
    def _fill_text_gaps(self, primary_text: str, all_results: List[Dict]) -> str:
        """Fill gaps in primary text using other OCR results."""
        # This is a simplified implementation
        # In production, you might use more sophisticated text alignment
        
        # If primary text is very short, try to find a better alternative
        if len(primary_text.strip()) < 100:
            for result in all_results:
                text = result.get('text', '')
                if len(text.strip()) > len(primary_text.strip()):
                    logger.info(f"Switching to longer text from {result.get('engine', 'unknown')}")
                    return text
        
        return primary_text
    
    def process_text(self, text: str) -> Dict:
        """Process text with full spaCy pipeline."""
        if not self.is_available():
            return self._fallback_processing(text)
        
        try:
            # Process with spaCy
            doc = self.nlp(text)
            
            # Extract sentences
            sentences = self._extract_sentences(doc)
            
            # Extract entities
            entities = self._extract_entities(doc)
            
            # Analyze structure
            structure = self._analyze_structure(doc)
            
            return {
                'sentences': sentences,
                'entities': entities,
                'structure': structure,
                'doc': doc,
                'tokens': [token.text for token in doc],
                'pos_tags': [(token.text, token.pos_) for token in doc],
                'metadata': {
                    'sentence_count': len(sentences),
                    'entity_count': len(entities),
                    'token_count': len(doc),
                    'char_count': len(text)
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing text with spaCy: {e}")
            return self._fallback_processing(text)
    
    def _extract_sentences(self, doc) -> List[Sentence]:
        """Extract sentences with metadata."""
        sentences = []
        
        for sent in doc.sents:
            # Extract tokens for this sentence
            tokens = [token.text for token in sent]
            
            # Extract entities in this sentence
            sentence_entities = []
            for ent in sent.ents:
                sentence_entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char - sent.start_char,
                    'end': ent.end_char - sent.start_char
                })
            
            sentence = Sentence(
                text=sent.text.strip(),
                start_char=sent.start_char,
                end_char=sent.end_char,
                tokens=tokens,
                entities=sentence_entities,
                metadata={
                    'length': len(sent.text),
                    'token_count': len(tokens),
                    'entity_count': len(sentence_entities)
                }
            )
            
            sentences.append(sentence)
        
        return sentences
    
    def _extract_entities(self, doc) -> List[Entity]:
        """Extract named entities."""
        entities = []
        
        for ent in doc.ents:
            entity = Entity(
                text=ent.text,
                label=ent.label_,
                start_char=ent.start_char,
                end_char=ent.end_char,
                confidence=1.0  # spaCy doesn't provide confidence scores by default
            )
            entities.append(entity)
        
        return entities
    
    def _analyze_structure(self, doc) -> DocumentStructure:
        """Analyze document structure."""
        # Identify potential section headers
        sections = self._identify_sections(doc)
        
        # Create structure analysis
        structure = DocumentStructure(
            sentences=self._extract_sentences(doc),
            entities=self._extract_entities(doc),
            sections=sections,
            metadata={
                'total_length': len(doc.text),
                'sentence_count': len(list(doc.sents)),
                'entity_count': len(doc.ents),
                'section_count': len(sections)
            }
        )
        
        return structure
    
    def _identify_sections(self, doc) -> List[Dict]:
        """Identify potential document sections."""
        sections = []
        
        # Look for patterns that indicate section headers
        section_patterns = [
            r'^[A-Z][A-Z\s]+:',  # ALL CAPS headers with colon
            r'^\d+\.\s+[A-Z]',   # Numbered sections
            r'^[IVX]+\.\s+',     # Roman numeral sections
            r'^SECTION\s+\d+',   # "SECTION 1" type headers
        ]
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            
            # Check if sentence matches section patterns
            found_pattern = False
            for pattern in section_patterns:
                if re.match(pattern, sent_text):
                    sections.append({
                        'text': sent_text,
                        'start_char': sent.start_char,
                        'end_char': sent.end_char,
                        'type': 'header',
                        'level': self._estimate_header_level(sent_text)
                    })
                    found_pattern = True
                    break
            
            # Also look for sentences that are short and title-case
            if (not found_pattern and len(sent_text) < 80 and 
                  sent_text.istitle() and 
                  not sent_text.endswith('.')):
                sections.append({
                    'text': sent_text,
                    'start_char': sent.start_char,
                    'end_char': sent.end_char,
                    'type': 'potential_header',
                    'level': 2
                })
        
        return sections
    
    def _estimate_header_level(self, text: str) -> int:
        """Estimate the hierarchical level of a header."""
        text = text.strip()
        
        # Level 1: All caps, short
        if text.isupper() and len(text) < 50:
            return 1
        
        # Level 1: Numbered with single digit
        if re.match(r'^\d+\.\s+', text):
            number = int(re.match(r'^(\d+)\.', text).group(1))
            return 1 if number < 10 else 2
        
        # Level 2: Roman numerals or title case
        if re.match(r'^[IVX]+\.\s+', text) or text.istitle():
            return 2
        
        # Default level
        return 3
    
    def segment_sentences(self, text: str) -> List[str]:
        """Segment text into sentences."""
        if not self.is_available():
            return self._fallback_sentence_segmentation(text)
        
        try:
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        except Exception as e:
            logger.error(f"Error in sentence segmentation: {e}")
            return self._fallback_sentence_segmentation(text)
    
    def extract_entities(self, text: str) -> List[Dict]:
        """Extract named entities from text."""
        if not self.is_available():
            return []
        
        try:
            doc = self.nlp(text)
            entities = []
            
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'description': spacy.explain(ent.label_),
                    'start': ent.start_char,
                    'end': ent.end_char
                })
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    def analyze_tokens(self, text: str) -> List[Dict]:
        """Analyze tokens with POS tags and dependencies."""
        if not self.is_available():
            return []
        
        try:
            doc = self.nlp(text)
            tokens = []
            
            for token in doc:
                tokens.append({
                    'text': token.text,
                    'lemma': token.lemma_,
                    'pos': token.pos_,
                    'tag': token.tag_,
                    'dep': token.dep_,
                    'is_alpha': token.is_alpha,
                    'is_stop': token.is_stop,
                    'is_punct': token.is_punct
                })
            
            return tokens
            
        except Exception as e:
            logger.error(f"Error analyzing tokens: {e}")
            return []
    
    def _fallback_processing(self, text: str) -> Dict:
        """Fallback processing when spaCy is not available."""
        sentences = self._fallback_sentence_segmentation(text)
        
        return {
            'sentences': [
                Sentence(
                    text=sent,
                    start_char=0,
                    end_char=len(sent),
                    tokens=sent.split(),
                    entities=[],
                    metadata={'length': len(sent)}
                ) for sent in sentences
            ],
            'entities': [],
            'structure': None,
            'doc': None,
            'tokens': text.split(),
            'pos_tags': [],
            'metadata': {
                'sentence_count': len(sentences),
                'entity_count': 0,
                'token_count': len(text.split()),
                'char_count': len(text),
                'fallback_mode': True
            }
        }
    
    def _fallback_sentence_segmentation(self, text: str) -> List[str]:
        """Basic sentence segmentation without spaCy."""
        # Simple regex-based sentence splitting
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def batch_process(self, texts: List[str], batch_size: int = 100) -> List[Dict]:
        """Process multiple texts efficiently."""
        if not self.is_available():
            return [self._fallback_processing(text) for text in texts]
        
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                # Process batch with spaCy's pipe for efficiency
                docs = list(self.nlp.pipe(batch, disable=['ner'], batch_size=batch_size))
                
                for doc in docs:
                    result = {
                        'sentences': [sent.text.strip() for sent in doc.sents],
                        'entities': [
                            {
                                'text': ent.text,
                                'label': ent.label_,
                                'start': ent.start_char,
                                'end': ent.end_char
                            }
                            for ent in doc.ents
                        ],
                        'tokens': [token.text for token in doc],
                        'metadata': {
                            'sentence_count': len(list(doc.sents)),
                            'entity_count': len(doc.ents),
                            'token_count': len(doc)
                        }
                    }
                    results.append(result)
                    
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                # Fallback to individual processing
                for text in batch:
                    results.append(self._fallback_processing(text))
        
        return results