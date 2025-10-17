"""
Phrase Matcher - Stage 4 of PDF Chunking Pipeline

Uses spaCy's PhraseMatcher to detect procurement-specific patterns:
- Section headers (Budget, Timeline, Department)
- Procurement terms (Public Bidding, Negotiated)
- Financial terms (MOOE, CO, Budget categories)
- Department codes (NCHFD, NCDPC, etc.)
- Timeline terms (Pre-procurement, Bid Evaluation, etc.)
"""

import logging
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum

try:
    import spacy
    from spacy.matcher import PhraseMatcher
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available for PhraseMatcher")

logger = logging.getLogger(__name__)

class ContentType(Enum):
    """Content type classifications."""
    SECTION_HEADER = "section_header"
    PROCUREMENT_MODE = "procurement_mode"
    FINANCIAL_TERM = "financial_term"
    DEPARTMENT_CODE = "department_code"
    TIMELINE_TERM = "timeline_term"
    BUDGET_CATEGORY = "budget_category"
    DOCUMENT_TYPE = "document_type"
    FISCAL_TERM = "fiscal_term"
    UNKNOWN = "unknown"

@dataclass
class PatternMatch:
    """Represents a pattern match."""
    text: str
    label: str
    start_char: int
    end_char: int
    content_type: ContentType
    confidence: float
    context: str = ""

class ProcurementPhraseMatcher:
    """Procurement-specific phrase matcher using spaCy."""
    
    def __init__(self, nlp=None):
        self.nlp = nlp
        self.matcher = None
        self.patterns = {}
        self._initialize_matcher()
    
    def _initialize_matcher(self):
        """Initialize the phrase matcher with patterns."""
        if not SPACY_AVAILABLE:
            logger.error("spaCy not available for phrase matching")
            return
        
        if self.nlp is None:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.error("spaCy model not found. Using basic English.")
                self.nlp = spacy.blank("en")
        
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self._setup_patterns()
        logger.info(f"Initialized PhraseMatcher with {len(self.patterns)} pattern categories")
    
    def is_available(self) -> bool:
        """Check if phrase matcher is available."""
        return SPACY_AVAILABLE and self.matcher is not None
    
    def _setup_patterns(self):
        """Set up all procurement-specific patterns."""
        
        # 1. Section Headers
        section_patterns = [
            "annual procurement plan",
            "procurement plan",
            "budget breakdown",
            "estimated budget",
            "total budget",
            "mode of procurement",
            "procurement mode",
            "timeline and schedule",
            "procurement timeline",
            "schedule of procurement",
            "implementing unit",
            "responsible unit",
            "procurement management unit",
            "bac composition",
            "bid evaluation committee",
            "technical working group",
            "end user unit",
            "requisitioning unit"
        ]
        self._add_patterns("SECTION_HEADER", section_patterns, ContentType.SECTION_HEADER)
        
        # 2. Procurement Modes/Methods
        procurement_mode_patterns = [
            "public bidding",
            "competitive bidding",
            "open competitive bidding",
            "restricted bidding",
            "negotiated procurement",
            "two stage bidding",
            "limited source bidding",
            "direct contracting",
            "repeat order",
            "shopping",
            "small value procurement",
            "emergency procurement",
            "lease",
            "scientific cooperation",
            "highly technical procurement"
        ]
        self._add_patterns("PROCUREMENT_MODE", procurement_mode_patterns, ContentType.PROCUREMENT_MODE)
        
        # 3. Financial/Budget Terms
        financial_patterns = [
            "mooe",
            "maintenance and other operating expenses",
            "capital outlay",
            "co",
            "ps",
            "personal services",
            "total approved budget",
            "approved budget for the contract",
            "abc",
            "estimated budget",
            "contract amount",
            "winning bid",
            "bid amount",
            "total contract cost",
            "budget allocation",
            "allotment",
            "fund source",
            "funding source"
        ]
        self._add_patterns("FINANCIAL_TERM", financial_patterns, ContentType.FINANCIAL_TERM)
        
        # 4. Department/Agency Codes
        department_patterns = [
            "nchfd",
            "national commission on human and family development",
            "ncdpc", 
            "national commission on disability and persons with disabilities",
            "ncpam",
            "national commission on population and migration",
            "ncmf",
            "national commission on muslim filipinos",
            "ncip",
            "national commission on indigenous peoples",
            "ncca",
            "national commission for culture and the arts",
            "ncw",
            "national commission on the role of filipino women",
            "nyc",
            "national youth commission",
            "nsc",
            "national senior citizens commission",
            "dilg",
            "department of the interior and local government",
            "dof",
            "department of finance",
            "dbm",
            "department of budget and management",
            "doh",
            "department of health",
            "deped",
            "department of education",
            "da",
            "department of agriculture",
            "denr",
            "department of environment and natural resources",
            "dpwh",
            "department of public works and highways",
            "dot",
            "department of tourism",
            "dtr",
            "department of transportation",
            "doe",
            "department of energy",
            "dti",
            "department of trade and industry",
            "dole",
            "department of labor and employment",
            "dswd",
            "department of social welfare and development",
            "dar",
            "department of agrarian reform",
            "dnd",
            "department of national defense",
            "dfa",
            "department of foreign affairs",
            "doj",
            "department of justice",
            "dict",
            "department of information and communications technology"
        ]
        self._add_patterns("DEPARTMENT_CODE", department_patterns, ContentType.DEPARTMENT_CODE)
        
        # 5. Timeline/Process Terms
        timeline_patterns = [
            "pre procurement conference",
            "pre-procurement conference",
            "advertisement",
            "posting",
            "submission and receipt of bids",
            "bid submission",
            "opening of bids",
            "bid opening",
            "preliminary examination of bids",
            "detailed evaluation of bids",
            "bid evaluation",
            "post qualification",
            "post-qualification",
            "award",
            "notice of award",
            "noa",
            "contract signing",
            "notice to proceed",
            "ntp",
            "delivery",
            "completion",
            "acceptance",
            "payment",
            "warranty period",
            "performance security",
            "bid security",
            "philgeps posting",
            "philgeps",
            "procurement monitoring report",
            "pmr"
        ]
        self._add_patterns("TIMELINE_TERM", timeline_patterns, ContentType.TIMELINE_TERM)
        
        # 6. Budget Categories
        budget_category_patterns = [
            "food supplies",
            "fuel oil and lubricants",
            "office supplies",
            "accountable forms",
            "non accountable forms",
            "drugs and medicines",
            "medical supplies",
            "textbooks",
            "instructional materials",
            "agricultural supplies",
            "other supplies and materials",
            "communication services",
            "internet subscription",
            "cable tv subscription",
            "telephone services",
            "postage and courier services",
            "electricity",
            "water",
            "professional services",
            "consultancy services",
            "janitorial services",
            "security services",
            "repairs and maintenance",
            "transportation and delivery",
            "rent",
            "training expenses",
            "office equipment",
            "information and communication technology equipment",
            "ict equipment",
            "technical and scientific equipment",
            "furniture and fixtures",
            "construction supplies",
            "motor vehicles",
            "heavy equipment"
        ]
        self._add_patterns("BUDGET_CATEGORY", budget_category_patterns, ContentType.BUDGET_CATEGORY)
        
        # 7. Document Types
        document_type_patterns = [
            "annual procurement plan",
            "app",
            "supplemental procurement plan",
            "spp",
            "procurement monitoring report",
            "pmr",
            "abstract of bids",
            "bid evaluation report",
            "ber",
            "resolution",
            "notice of award",
            "noa",
            "purchase order",
            "po",
            "contract",
            "work order",
            "job order",
            "inspection and acceptance report",
            "iar",
            "certificate of inspection",
            "performance evaluation report",
            "accomplishment report"
        ]
        self._add_patterns("DOCUMENT_TYPE", document_type_patterns, ContentType.DOCUMENT_TYPE)
        
        # 8. Fiscal/Calendar Terms
        fiscal_patterns = [
            "fiscal year",
            "fy",
            "calendar year",
            "cy",
            "quarter",
            "q1", "q2", "q3", "q4",
            "first quarter",
            "second quarter", 
            "third quarter",
            "fourth quarter",
            "semester",
            "first semester",
            "second semester",
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
            "jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec"
        ]
        self._add_patterns("FISCAL_TERM", fiscal_patterns, ContentType.FISCAL_TERM)
    
    def _add_patterns(self, label: str, patterns: List[str], content_type: ContentType):
        """Add patterns to the matcher."""
        if not self.is_available():
            return
        
        # Convert patterns to Doc objects
        pattern_docs = [self.nlp(pattern) for pattern in patterns]
        
        # Add to matcher
        self.matcher.add(label, pattern_docs)
        
        # Store pattern metadata
        self.patterns[label] = {
            'patterns': patterns,
            'content_type': content_type,
            'count': len(patterns)
        }
        
        logger.debug(f"Added {len(patterns)} patterns for {label}")
    
    def find_matches(self, doc) -> List[PatternMatch]:
        """Find all pattern matches in a document."""
        if not self.is_available():
            return []
        
        matches = self.matcher(doc)
        pattern_matches = []
        
        for match_id, start, end in matches:
            label = self.nlp.vocab.strings[match_id]
            span = doc[start:end]
            
            # Get content type
            content_type = self.patterns.get(label, {}).get('content_type', ContentType.UNKNOWN)
            
            # Calculate confidence (simple heuristic based on exact match)
            confidence = 1.0  # Exact phrase matches get full confidence
            
            # Get context (surrounding text)
            context_start = max(0, start - 5)
            context_end = min(len(doc), end + 5)
            context = doc[context_start:context_end].text
            
            match = PatternMatch(
                text=span.text,
                label=label,
                start_char=span.start_char,
                end_char=span.end_char,
                content_type=content_type,
                confidence=confidence,
                context=context
            )
            
            pattern_matches.append(match)
        
        return pattern_matches
    
    def find_all_matches(self, text: str) -> List[PatternMatch]:
        """Find matches in raw text."""
        if not self.is_available():
            return []
        
        try:
            doc = self.nlp(text)
            return self.find_matches(doc)
        except Exception as e:
            logger.error(f"Error finding matches: {e}")
            return []
    
    def tag_content_type(self, sentence: str) -> ContentType:
        """Tag the content type of a sentence."""
        matches = self.find_all_matches(sentence)
        
        if not matches:
            return ContentType.UNKNOWN
        
        # Return the content type of the first (most confident) match
        return matches[0].content_type
    
    def extract_section_hierarchy(self, text: str) -> Dict:
        """Extract document section hierarchy."""
        if not self.is_available():
            return {}
        
        try:
            doc = self.nlp(text)
            matches = self.find_matches(doc)
            
            # Group matches by content type
            hierarchy = {content_type.value: [] for content_type in ContentType}
            
            for match in matches:
                hierarchy[match.content_type.value].append({
                    'text': match.text,
                    'position': match.start_char,
                    'confidence': match.confidence
                })
            
            # Sort by position
            for content_type in hierarchy:
                hierarchy[content_type].sort(key=lambda x: x['position'])
            
            return hierarchy
            
        except Exception as e:
            logger.error(f"Error extracting hierarchy: {e}")
            return {}
    
    def get_pattern_statistics(self) -> Dict:
        """Get statistics about loaded patterns."""
        if not self.patterns:
            return {}
        
        stats = {
            'total_categories': len(self.patterns),
            'total_patterns': sum(p['count'] for p in self.patterns.values()),
            'categories': {}
        }
        
        for label, pattern_info in self.patterns.items():
            stats['categories'][label] = {
                'pattern_count': pattern_info['count'],
                'content_type': pattern_info['content_type'].value,
                'sample_patterns': pattern_info['patterns'][:5]  # First 5 as examples
            }
        
        return stats
    
    def match_statistics(self, text: str) -> Dict:
        """Get statistics about matches found in text."""
        matches = self.find_all_matches(text)
        
        if not matches:
            return {'total_matches': 0}
        
        # Count by content type
        content_type_counts = {}
        for match in matches:
            content_type = match.content_type.value
            content_type_counts[content_type] = content_type_counts.get(content_type, 0) + 1
        
        # Count by label
        label_counts = {}
        for match in matches:
            label_counts[match.label] = label_counts.get(match.label, 0) + 1
        
        return {
            'total_matches': len(matches),
            'unique_labels': len(set(match.label for match in matches)),
            'content_type_distribution': content_type_counts,
            'label_distribution': label_counts,
            'average_confidence': sum(match.confidence for match in matches) / len(matches),
            'matches': [
                {
                    'text': match.text,
                    'label': match.label,
                    'content_type': match.content_type.value,
                    'confidence': match.confidence
                } for match in matches[:10]  # First 10 matches
            ]
        }
    
    def get_procurement_entities(self, text: str) -> Dict:
        """Extract procurement-specific entities and their relationships."""
        matches = self.find_all_matches(text)
        
        entities = {
            'departments': [],
            'procurement_modes': [],
            'budget_terms': [],
            'timeline_items': [],
            'document_types': []
        }
        
        for match in matches:
            if match.content_type == ContentType.DEPARTMENT_CODE:
                entities['departments'].append(match.text)
            elif match.content_type == ContentType.PROCUREMENT_MODE:
                entities['procurement_modes'].append(match.text)
            elif match.content_type == ContentType.FINANCIAL_TERM:
                entities['budget_terms'].append(match.text)
            elif match.content_type == ContentType.TIMELINE_TERM:
                entities['timeline_items'].append(match.text)
            elif match.content_type == ContentType.DOCUMENT_TYPE:
                entities['document_types'].append(match.text)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities