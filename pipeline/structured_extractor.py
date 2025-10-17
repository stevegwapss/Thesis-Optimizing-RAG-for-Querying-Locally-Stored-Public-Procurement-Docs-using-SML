"""
Structured Extractor - Stage 5 of PDF Chunking Pipeline

Uses Python data structures + regex + PDFPlumber for precise extraction:
- Currency amounts (PHP X,XXX,XXX.XX)
- Dates (MM/DD/YYYY, various formats)
- Department codes (NCHFD-DEPT)
- Budget categories (MOOE, CO)
- Table structures with PDFPlumber
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from datetime import datetime, date
import calendar

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logging.warning("PDFPlumber not available")

logger = logging.getLogger(__name__)

@dataclass
class ExtractedAmount:
    """Represents an extracted currency amount."""
    raw_text: str
    amount: Decimal
    currency: str
    position: int
    confidence: float

@dataclass
class ExtractedDate:
    """Represents an extracted date."""
    raw_text: str
    parsed_date: Optional[date]
    date_format: str
    position: int
    confidence: float

@dataclass
class ExtractedCode:
    """Represents an extracted code (department, etc.)."""
    raw_text: str
    code_type: str
    normalized_code: str
    position: int
    confidence: float

@dataclass
class StructuredTable:
    """Represents a structured table."""
    id: str
    headers: List[str]
    rows: List[Dict[str, str]]
    metadata: Dict
    financial_data: List[ExtractedAmount]
    date_data: List[ExtractedDate]

class StructuredExtractor:
    """Extracts structured data using regex patterns and PDFPlumber."""
    
    def __init__(self):
        self._compile_patterns()
        self.supported_currencies = ['PHP', 'USD', 'EUR', 'PESO', 'PESOS']
        self.fiscal_years = list(range(2010, 2030))  # Reasonable range for procurement docs
    
    def _compile_patterns(self):
        """Compile all regex patterns for efficiency."""
        
        # Currency patterns - matches various PHP amount formats
        self.currency_patterns = {
            'php_standard': re.compile(
                r'PHP\s*([\d,]+\.?\d*)',
                re.IGNORECASE
            ),
            'php_with_centavos': re.compile(
                r'PHP\s*([\d,]+\.\d{2})',
                re.IGNORECASE
            ),
            'peso_word': re.compile(
                r'(?:pesos?|peso)\s*([\d,]+\.?\d*)',
                re.IGNORECASE
            ),
            'amount_only': re.compile(
                r'(?:^|\s)([\d,]+\.\d{2})(?=\s|$)'
            ),
            'whole_amount': re.compile(
                r'(?:^|\s)([\d,]{4,})(?=\s|$)'  # Large numbers with commas
            )
        }
        
        # Date patterns - various date formats common in procurement docs
        self.date_patterns = {
            'mm_dd_yyyy': re.compile(
                r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})'
            ),
            'dd_mm_yyyy': re.compile(
                r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})'
            ),
            'month_dd_yyyy': re.compile(
                r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})',
                re.IGNORECASE
            ),
            'dd_month_yyyy': re.compile(
                r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})',
                re.IGNORECASE
            ),
            'week_month_year': re.compile(
                r'(\d{1,2})(?:st|nd|rd|th)?\s+week\s+(\w+)\s+(\d{4})',
                re.IGNORECASE
            ),
            'quarter_year': re.compile(
                r'(Q[1-4]|(?:First|Second|Third|Fourth)\s+Quarter)\s+(\d{4})',
                re.IGNORECASE
            ),
            'fy_year': re.compile(
                r'FY\s*(\d{4})',
                re.IGNORECASE
            ),
            'fiscal_year': re.compile(
                r'Fiscal\s+Year\s+(\d{4})',
                re.IGNORECASE
            )
        }
        
        # Department/Agency code patterns
        self.code_patterns = {
            'department_code': re.compile(
                r'\b([A-Z]{2,}(?:-[A-Z]{2,})*)\b'
            ),
            'office_code': re.compile(
                r'\b([A-Z]{2,}\d+)\b'
            ),
            'unit_code': re.compile(
                r'\b(Unit\s+[A-Z0-9]+)\b',
                re.IGNORECASE
            ),
            'division_code': re.compile(
                r'\b(Div\.|Division)\s+([A-Z0-9]+)\b',
                re.IGNORECASE
            )
        }
        
        # Budget category patterns
        self.budget_patterns = {
            'budget_type': re.compile(
                r'\b(MOOE|CO|PS|Capital\s+Outlay|Personal\s+Services|Maintenance\s+and\s+Other\s+Operating\s+Expenses)\b',
                re.IGNORECASE
            ),
            'fund_source': re.compile(
                r'\b(General\s+Fund|Special\s+Fund|GAA|General\s+Appropriations\s+Act)\b',
                re.IGNORECASE
            )
        }
        
        # Procurement-specific patterns
        self.procurement_patterns = {
            'contract_number': re.compile(
                r'\b(?:Contract|PO|Purchase\s+Order)\s+(?:No\.?\s*)?([A-Z0-9\-]+)\b',
                re.IGNORECASE
            ),
            'bid_number': re.compile(
                r'\b(?:Bid|Bidding)\s+(?:No\.?\s*)?([A-Z0-9\-]+)\b',
                re.IGNORECASE
            )
        }
    
    def extract_currency_amounts(self, text: str) -> List[ExtractedAmount]:
        """Extract currency amounts from text."""
        amounts = []
        
        for pattern_name, pattern in self.currency_patterns.items():
            for match in pattern.finditer(text):
                try:
                    # Extract the amount string
                    if pattern_name in ['php_standard', 'php_with_centavos']:
                        amount_str = match.group(1)
                        currency = 'PHP'
                        raw_text = match.group(0)
                    elif pattern_name == 'peso_word':
                        amount_str = match.group(1)
                        currency = 'PHP'
                        raw_text = match.group(0)
                    else:
                        amount_str = match.group(1)
                        currency = 'PHP'  # Assume PHP for amounts without explicit currency
                        raw_text = match.group(0)
                    
                    # Clean and convert amount
                    clean_amount = amount_str.replace(',', '')
                    decimal_amount = Decimal(clean_amount)
                    
                    # Calculate confidence based on pattern specificity
                    confidence = self._calculate_amount_confidence(pattern_name, raw_text)
                    
                    # Only include reasonable amounts (not single digits, not impossibly large)
                    if 10 <= decimal_amount <= 1000000000000:  # 10 PHP to 1 trillion PHP
                        amounts.append(ExtractedAmount(
                            raw_text=raw_text,
                            amount=decimal_amount,
                            currency=currency,
                            position=match.start(),
                            confidence=confidence
                        ))
                        
                except (ValueError, InvalidOperation) as e:
                    logger.debug(f"Could not parse amount '{match.group(0)}': {e}")
                    continue
        
        # Remove duplicates and sort by position
        amounts = self._deduplicate_amounts(amounts)
        amounts.sort(key=lambda x: x.position)
        
        return amounts
    
    def extract_dates(self, text: str) -> List[ExtractedDate]:
        """Extract dates from text."""
        dates = []
        
        for pattern_name, pattern in self.date_patterns.items():
            for match in pattern.finditer(text):
                try:
                    parsed_date = self._parse_date_match(pattern_name, match)
                    confidence = self._calculate_date_confidence(pattern_name, match.group(0))
                    
                    if parsed_date:
                        dates.append(ExtractedDate(
                            raw_text=match.group(0),
                            parsed_date=parsed_date,
                            date_format=pattern_name,
                            position=match.start(),
                            confidence=confidence
                        ))
                        
                except Exception as e:
                    logger.debug(f"Could not parse date '{match.group(0)}': {e}")
                    continue
        
        # Remove duplicates and sort by position
        dates = self._deduplicate_dates(dates)
        dates.sort(key=lambda x: x.position)
        
        return dates
    
    def extract_department_codes(self, text: str) -> List[ExtractedCode]:
        """Extract department and organizational codes."""
        codes = []
        
        # Known department codes for higher confidence
        known_departments = {
            'NCHFD', 'NCDPC', 'NCPAM', 'NCMF', 'NCIP', 'NCCA', 'NCW', 'NYC', 'NSC',
            'DILG', 'DOF', 'DBM', 'DOH', 'DEPED', 'DA', 'DENR', 'DPWH', 'DOT', 'DTR',
            'DOE', 'DTI', 'DOLE', 'DSWD', 'DAR', 'DND', 'DFA', 'DOJ', 'DICT'
        }
        
        for pattern_name, pattern in self.code_patterns.items():
            for match in pattern.finditer(text):
                try:
                    if pattern_name == 'department_code':
                        code = match.group(1)
                        code_type = 'department'
                    elif pattern_name == 'office_code':
                        code = match.group(1)
                        code_type = 'office'
                    elif pattern_name == 'unit_code':
                        code = match.group(1)
                        code_type = 'unit'
                    elif pattern_name == 'division_code':
                        code = match.group(2)
                        code_type = 'division'
                    else:
                        code = match.group(1)
                        code_type = 'unknown'
                    
                    # Calculate confidence
                    confidence = 0.5  # Base confidence
                    if code.upper() in known_departments:
                        confidence = 0.9
                    elif len(code) >= 3 and code.isupper():
                        confidence = 0.7
                    
                    codes.append(ExtractedCode(
                        raw_text=match.group(0),
                        code_type=code_type,
                        normalized_code=code.upper(),
                        position=match.start(),
                        confidence=confidence
                    ))
                    
                except Exception as e:
                    logger.debug(f"Could not parse code '{match.group(0)}': {e}")
                    continue
        
        # Remove duplicates and sort by confidence, then position
        codes = self._deduplicate_codes(codes)
        codes.sort(key=lambda x: (-x.confidence, x.position))
        
        return codes
    
    def extract_budget_categories(self, text: str) -> List[ExtractedCode]:
        """Extract budget categories and fund sources."""
        categories = []
        
        for pattern_name, pattern in self.budget_patterns.items():
            for match in pattern.finditer(text):
                try:
                    category = match.group(1)
                    
                    # Normalize category names
                    normalized = self._normalize_budget_category(category)
                    
                    categories.append(ExtractedCode(
                        raw_text=match.group(0),
                        code_type=pattern_name,
                        normalized_code=normalized,
                        position=match.start(),
                        confidence=0.8  # High confidence for budget patterns
                    ))
                    
                except Exception as e:
                    logger.debug(f"Could not parse budget category '{match.group(0)}': {e}")
                    continue
        
        return categories
    
    def extract_tables_with_pdfplumber(self, pdf_path: Path) -> List[StructuredTable]:
        """Extract and structure tables using PDFPlumber."""
        if not PDFPLUMBER_AVAILABLE:
            logger.error("PDFPlumber not available for table extraction")
            return []
        
        tables = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = self._extract_page_tables(page, page_num)
                    tables.extend(page_tables)
                    
        except Exception as e:
            logger.error(f"Error extracting tables from {pdf_path}: {e}")
        
        return tables
    
    def _extract_page_tables(self, page, page_num: int) -> List[StructuredTable]:
        """Extract tables from a single page."""
        tables = []
        
        try:
            raw_tables = page.extract_tables()
            
            for table_idx, raw_table in enumerate(raw_tables):
                if raw_table and len(raw_table) > 1:
                    structured_table = self._structure_table(raw_table, page_num, table_idx)
                    if structured_table:
                        tables.append(structured_table)
                        
        except Exception as e:
            logger.warning(f"Error extracting tables from page {page_num}: {e}")
        
        return tables
    
    def _structure_table(self, raw_table: List[List[str]], page_num: int, table_idx: int) -> Optional[StructuredTable]:
        """Convert raw table to structured format with extracted data."""
        try:
            if not raw_table or len(raw_table) < 2:
                return None
            
            # Extract headers
            headers = [self._clean_cell_text(cell) for cell in raw_table[0]]
            headers = [h or f"col_{i}" for i, h in enumerate(headers)]
            
            # Extract rows
            rows = []
            all_financial_data = []
            all_date_data = []
            
            for row_data in raw_table[1:]:
                row_dict = {}
                row_text = ""
                
                for i, cell in enumerate(row_data):
                    if i < len(headers):
                        clean_cell = self._clean_cell_text(cell)
                        row_dict[headers[i]] = clean_cell
                        row_text += " " + clean_cell
                
                # Only add non-empty rows
                if any(value.strip() for value in row_dict.values() if value):
                    rows.append(row_dict)
                    
                    # Extract financial data from row text
                    row_amounts = self.extract_currency_amounts(row_text)
                    all_financial_data.extend(row_amounts)
                    
                    # Extract dates from row text
                    row_dates = self.extract_dates(row_text)
                    all_date_data.extend(row_dates)
            
            if not rows:
                return None
            
            # Determine table type and metadata
            table_metadata = self._analyze_table_content(headers, rows)
            
            table_id = f"page_{page_num}_table_{table_idx}"
            
            return StructuredTable(
                id=table_id,
                headers=headers,
                rows=rows,
                metadata={
                    **table_metadata,
                    'page_number': page_num,
                    'table_index': table_idx,
                    'row_count': len(rows),
                    'col_count': len(headers)
                },
                financial_data=all_financial_data,
                date_data=all_date_data
            )
            
        except Exception as e:
            logger.warning(f"Error structuring table: {e}")
            return None
    
    def _clean_cell_text(self, cell: Optional[str]) -> str:
        """Clean and normalize cell text."""
        if not cell:
            return ""
        
        # Remove extra whitespace and normalize
        cleaned = re.sub(r'\s+', ' ', str(cell).strip())
        
        # Remove common artifacts
        cleaned = cleaned.replace('\n', ' ').replace('\t', ' ')
        
        return cleaned
    
    def _analyze_table_content(self, headers: List[str], rows: List[Dict]) -> Dict:
        """Analyze table content to determine type and characteristics."""
        metadata = {
            'table_type': 'data',
            'has_financial_data': False,
            'has_dates': False,
            'likely_budget_table': False,
            'likely_timeline_table': False
        }
        
        # Analyze headers
        header_text = " ".join(headers).lower()
        
        # Financial indicators
        financial_keywords = ['amount', 'cost', 'budget', 'price', 'php', 'peso', 'total']
        if any(keyword in header_text for keyword in financial_keywords):
            metadata['has_financial_data'] = True
            metadata['likely_budget_table'] = True
        
        # Date/timeline indicators  
        date_keywords = ['date', 'week', 'month', 'quarter', 'schedule', 'timeline']
        if any(keyword in header_text for keyword in date_keywords):
            metadata['has_dates'] = True
            metadata['likely_timeline_table'] = True
        
        # Procurement indicators
        procurement_keywords = ['procurement', 'bid', 'contract', 'supplier', 'mode']
        if any(keyword in header_text for keyword in procurement_keywords):
            metadata['table_type'] = 'procurement'
        
        # Sample some row content for additional analysis
        sample_rows = rows[:5]  # First 5 rows
        sample_text = " ".join([
            " ".join(row.values()) for row in sample_rows
        ]).lower()
        
        # Check for common procurement terms in content
        if any(term in sample_text for term in ['public bidding', 'negotiated', 'shopping']):
            metadata['table_type'] = 'procurement'
        
        return metadata
    
    def build_structured_document(self, text: str, tables: List[StructuredTable]) -> Dict:
        """Build a comprehensive structured document representation."""
        # Extract all structured data
        amounts = self.extract_currency_amounts(text)
        dates = self.extract_dates(text)
        dept_codes = self.extract_department_codes(text)
        budget_categories = self.extract_budget_categories(text)
        
        # Analyze text structure
        sections = self._identify_text_sections(text)
        
        # Calculate summary statistics
        total_amount = sum(amount.amount for amount in amounts)
        
        # Determine document characteristics
        doc_characteristics = self._analyze_document_type(text, amounts, dates, dept_codes)
        
        structured_doc = {
            'metadata': {
                'total_text_length': len(text),
                'extraction_timestamp': datetime.now().isoformat(),
                'document_type': doc_characteristics.get('type', 'unknown'),
                'fiscal_year': doc_characteristics.get('fiscal_year'),
                'department': doc_characteristics.get('department'),
                'confidence_score': doc_characteristics.get('confidence', 0.0)
            },
            'financial_data': {
                'amounts': [
                    {
                        'raw_text': amt.raw_text,
                        'amount': float(amt.amount),
                        'currency': amt.currency,
                        'position': amt.position,
                        'confidence': amt.confidence
                    } for amt in amounts
                ],
                'total_amount': float(total_amount),
                'amount_count': len(amounts),
                'budget_categories': [
                    {
                        'raw_text': cat.raw_text,
                        'category': cat.normalized_code,
                        'type': cat.code_type,
                        'confidence': cat.confidence
                    } for cat in budget_categories
                ]
            },
            'temporal_data': {
                'dates': [
                    {
                        'raw_text': dt.raw_text,
                        'parsed_date': dt.parsed_date.isoformat() if dt.parsed_date else None,
                        'format': dt.date_format,
                        'confidence': dt.confidence
                    } for dt in dates
                ],
                'date_count': len(dates)
            },
            'organizational_data': {
                'departments': [
                    {
                        'raw_text': code.raw_text,
                        'code': code.normalized_code,
                        'type': code.code_type,
                        'confidence': code.confidence
                    } for code in dept_codes
                ],
                'department_count': len(dept_codes)
            },
            'tables': [
                {
                    'id': table.id,
                    'headers': table.headers,
                    'row_count': len(table.rows),
                    'col_count': len(table.headers),
                    'metadata': table.metadata,
                    'financial_data_count': len(table.financial_data),
                    'date_data_count': len(table.date_data)
                } for table in tables
            ],
            'text_structure': {
                'sections': sections,
                'section_count': len(sections)
            }
        }
        
        return structured_doc
    
    def _calculate_amount_confidence(self, pattern_name: str, raw_text: str) -> float:
        """Calculate confidence score for extracted amounts."""
        base_confidence = {
            'php_standard': 0.9,
            'php_with_centavos': 0.95,
            'peso_word': 0.8,
            'amount_only': 0.6,
            'whole_amount': 0.5
        }
        
        confidence = base_confidence.get(pattern_name, 0.5)
        
        # Boost confidence for amounts with proper formatting
        if ',' in raw_text:  # Has thousands separators
            confidence += 0.05
        if '.' in raw_text and raw_text.split('.')[-1].isdigit() and len(raw_text.split('.')[-1]) == 2:
            confidence += 0.05  # Has centavos
        
        return min(confidence, 1.0)
    
    def _calculate_date_confidence(self, pattern_name: str, raw_text: str) -> float:
        """Calculate confidence score for extracted dates."""
        base_confidence = {
            'mm_dd_yyyy': 0.8,
            'dd_mm_yyyy': 0.8,
            'month_dd_yyyy': 0.9,
            'dd_month_yyyy': 0.9,
            'week_month_year': 0.7,
            'quarter_year': 0.8,
            'fy_year': 0.9,
            'fiscal_year': 0.95
        }
        
        return base_confidence.get(pattern_name, 0.5)
    
    def _parse_date_match(self, pattern_name: str, match) -> Optional[date]:
        """Parse a date match based on the pattern type."""
        try:
            if pattern_name in ['mm_dd_yyyy', 'dd_mm_yyyy']:
                month, day, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
                return date(year, month, day)
            
            elif pattern_name in ['month_dd_yyyy', 'dd_month_yyyy']:
                if pattern_name == 'month_dd_yyyy':
                    month_name, day, year = match.group(1), int(match.group(2)), int(match.group(3))
                else:
                    day, month_name, year = int(match.group(1)), match.group(2), int(match.group(3))
                
                month = list(calendar.month_name).index(month_name.title())
                return date(year, month, day)
            
            elif pattern_name in ['fy_year', 'fiscal_year']:
                year = int(match.group(1))
                return date(year, 1, 1)  # Use January 1st as representative date
            
            # For quarter and week patterns, return approximate dates
            elif pattern_name == 'quarter_year':
                quarter_str, year = match.group(1), int(match.group(2))
                if 'Q1' in quarter_str or 'First' in quarter_str:
                    return date(year, 1, 1)
                elif 'Q2' in quarter_str or 'Second' in quarter_str:
                    return date(year, 4, 1)
                elif 'Q3' in quarter_str or 'Third' in quarter_str:
                    return date(year, 7, 1)
                elif 'Q4' in quarter_str or 'Fourth' in quarter_str:
                    return date(year, 10, 1)
            
            return None
            
        except (ValueError, IndexError):
            return None
    
    def _deduplicate_amounts(self, amounts: List[ExtractedAmount]) -> List[ExtractedAmount]:
        """Remove duplicate amounts keeping the highest confidence."""
        seen = {}
        for amount in amounts:
            key = (amount.amount, amount.currency)
            if key not in seen or amount.confidence > seen[key].confidence:
                seen[key] = amount
        return list(seen.values())
    
    def _deduplicate_dates(self, dates: List[ExtractedDate]) -> List[ExtractedDate]:
        """Remove duplicate dates keeping the highest confidence."""
        seen = {}
        for date_item in dates:
            if date_item.parsed_date:
                key = date_item.parsed_date
                if key not in seen or date_item.confidence > seen[key].confidence:
                    seen[key] = date_item
        return list(seen.values())
    
    def _deduplicate_codes(self, codes: List[ExtractedCode]) -> List[ExtractedCode]:
        """Remove duplicate codes keeping the highest confidence."""
        seen = {}
        for code in codes:
            key = (code.normalized_code, code.code_type)
            if key not in seen or code.confidence > seen[key].confidence:
                seen[key] = code
        return list(seen.values())
    
    def _normalize_budget_category(self, category: str) -> str:
        """Normalize budget category names."""
        category = category.upper().strip()
        
        # Common normalizations
        normalizations = {
            'MAINTENANCE AND OTHER OPERATING EXPENSES': 'MOOE',
            'CAPITAL OUTLAY': 'CO',
            'PERSONAL SERVICES': 'PS'
        }
        
        return normalizations.get(category, category)
    
    def _identify_text_sections(self, text: str) -> List[Dict]:
        """Identify sections in the text."""
        sections = []
        
        # Simple section identification based on common patterns
        section_patterns = [
            r'^[A-Z][A-Z\s]+:',  # ALL CAPS headers
            r'^\d+\.\s+[A-Z]',   # Numbered sections
            r'^SECTION\s+\d+',   # Section headers
        ]
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            for pattern in section_patterns:
                if re.match(pattern, line):
                    sections.append({
                        'text': line,
                        'line_number': i,
                        'type': 'header'
                    })
                    break
        
        return sections
    
    def _analyze_document_type(self, text: str, amounts: List[ExtractedAmount], 
                              dates: List[ExtractedDate], codes: List[ExtractedCode]) -> Dict:
        """Analyze document to determine type and characteristics."""
        text_lower = text.lower()
        
        # Document type indicators
        doc_type = 'unknown'
        confidence = 0.5
        
        if 'procurement plan' in text_lower:
            doc_type = 'procurement_plan'
            confidence = 0.9
        elif 'monitoring report' in text_lower:
            doc_type = 'monitoring_report'
            confidence = 0.9
        elif 'bid evaluation' in text_lower:
            doc_type = 'bid_evaluation'
            confidence = 0.8
        elif 'contract' in text_lower:
            doc_type = 'contract'
            confidence = 0.7
        
        # Extract fiscal year
        fiscal_year = None
        fy_pattern = re.search(r'(?:FY|Fiscal\s+Year)\s*(\d{4})', text, re.IGNORECASE)
        if fy_pattern:
            fiscal_year = int(fy_pattern.group(1))
        
        # Extract primary department
        department = None
        if codes:
            # Use the highest confidence department code
            dept_codes = [c for c in codes if c.code_type == 'department']
            if dept_codes:
                department = max(dept_codes, key=lambda x: x.confidence).normalized_code
        
        return {
            'type': doc_type,
            'confidence': confidence,
            'fiscal_year': fiscal_year,
            'department': department
        }