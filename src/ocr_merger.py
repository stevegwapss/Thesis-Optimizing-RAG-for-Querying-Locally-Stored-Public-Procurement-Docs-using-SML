"""
OCR Results Merger

This module implements intelligent merging of OCR results from multiple engines.
It uses confidence scoring and content analysis to select the best result for each
text region and creates a unified, high-quality output.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from difflib import SequenceMatcher
import statistics

from .models import OCRResult, OCREngine, RawContent, PDFType


logger = logging.getLogger(__name__)


class OCRMerger:
    """Merges OCR results from multiple engines using confidence scoring."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize OCR merger with configuration."""
        self.config = config or {}
        
        # Engine priority weights (can be adjusted based on document type)
        self.engine_weights = {
            OCREngine.TESSERACT: 1.0,
            OCREngine.PYPDF2: 1.0,
            OCREngine.PDFPLUMBER: 1.0
        }
        
        # Confidence thresholds
        self.min_confidence = self.config.get('min_confidence', 0.3)
        self.high_confidence = self.config.get('high_confidence', 0.8)
        
        # Text similarity threshold for cross-validation
        self.similarity_threshold = self.config.get('similarity_threshold', 0.7)
    
    def merge_ocr_results(self, raw_content: RawContent, pdf_type: PDFType) -> RawContent:
        """
        Merge OCR results from multiple engines into a single best result.
        
        Args:
            raw_content: RawContent with results from multiple OCR engines
            pdf_type: Type of PDF (digital, scanned, hybrid)
            
        Returns:
            Updated RawContent with merged text and best engine selection
        """
        ocr_results = raw_content.ocr_results
        
        if not ocr_results:
            logger.warning("No OCR results to merge")
            return raw_content
        
        # Filter successful results
        successful_results = [r for r in ocr_results if r.success and r.text.strip()]
        
        if not successful_results:
            logger.warning("No successful OCR results found")
            # Use the first result even if failed
            if ocr_results:
                raw_content.merged_text = ocr_results[0].text
                raw_content.best_engine = ocr_results[0].engine
                raw_content.merged_confidence = 0.0
            return raw_content
        
        # Adjust engine weights based on PDF type
        adjusted_weights = self._adjust_weights_for_pdf_type(pdf_type)
        
        # Choose best result using multiple strategies
        best_result = self._select_best_result(successful_results, adjusted_weights)
        
        # Enhance result with cross-validation
        enhanced_text, enhanced_confidence = self._enhance_with_cross_validation(
            best_result, successful_results
        )
        
        # Update raw content
        raw_content.merged_text = enhanced_text
        raw_content.best_engine = best_result.engine
        raw_content.merged_confidence = enhanced_confidence
        
        logger.debug(
            f"Merged OCR results: engine={best_result.engine}, "
            f"confidence={enhanced_confidence:.3f}, "
            f"text_length={len(enhanced_text)}"
        )
        
        return raw_content
    
    def _adjust_weights_for_pdf_type(self, pdf_type: PDFType) -> Dict[OCREngine, float]:
        """Adjust engine weights based on PDF type."""
        weights = self.engine_weights.copy()
        
        if pdf_type == PDFType.DIGITAL:
            # Digital PDFs: prefer PyPDF2 and PDFPlumber
            weights[OCREngine.PYPDF2] = 1.5
            weights[OCREngine.PDFPLUMBER] = 1.3
            weights[OCREngine.TESSERACT] = 0.7
            
        elif pdf_type == PDFType.SCANNED:
            # Scanned PDFs: prefer Tesseract
            weights[OCREngine.TESSERACT] = 1.5
            weights[OCREngine.PYPDF2] = 0.5
            weights[OCREngine.PDFPLUMBER] = 0.8
            
        elif pdf_type == PDFType.HYBRID:
            # Hybrid PDFs: balanced approach
            weights[OCREngine.TESSERACT] = 1.1
            weights[OCREngine.PYPDF2] = 1.2
            weights[OCREngine.PDFPLUMBER] = 1.3
        
        return weights
    
    def _select_best_result(
        self, 
        results: List[OCRResult], 
        weights: Dict[OCREngine, float]
    ) -> OCRResult:
        """Select the best OCR result using weighted scoring."""
        if len(results) == 1:
            return results[0]
        
        best_result = None
        best_score = -1
        
        for result in results:
            # Calculate weighted score
            engine_weight = weights.get(result.engine, 1.0)
            confidence_score = result.confidence
            text_quality_score = self._calculate_text_quality_score(result.text)
            
            # Combined score
            total_score = (
                confidence_score * 0.4 +
                text_quality_score * 0.4 +
                engine_weight * 0.2
            )
            
            logger.debug(
                f"Engine {result.engine}: confidence={confidence_score:.3f}, "
                f"quality={text_quality_score:.3f}, weight={engine_weight:.3f}, "
                f"total={total_score:.3f}"
            )
            
            if total_score > best_score:
                best_score = total_score
                best_result = result
        
        return best_result or results[0]
    
    def _calculate_text_quality_score(self, text: str) -> float:
        """Calculate text quality score based on various metrics."""
        if not text:
            return 0.0
        
        text = text.strip()
        if not text:
            return 0.0
        
        score = 0.5  # Base score
        
        # Length bonus (but not too long)
        length_score = min(1.0, len(text) / 1000)
        score += length_score * 0.2
        
        # Character diversity
        unique_chars = len(set(text.lower()))
        diversity_score = min(1.0, unique_chars / 50)
        score += diversity_score * 0.1
        
        # Word structure
        words = text.split()
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            word_length_score = min(1.0, avg_word_length / 6)  # Optimal around 6 chars
            score += word_length_score * 0.1
        
        # Alphabetic character ratio
        alpha_chars = sum(1 for c in text if c.isalpha())
        alpha_ratio = alpha_chars / len(text) if text else 0
        score += alpha_ratio * 0.2
        
        # Penalize common OCR errors
        error_patterns = [
            r'[���]+',  # Replacement characters
            r'\?{3,}',  # Multiple question marks
            r'[Il1]{5,}',  # Confused I/l/1 sequences
            r'\s{5,}',  # Excessive whitespace
        ]
        
        for pattern in error_patterns:
            matches = len(re.findall(pattern, text))
            score -= matches * 0.05
        
        # Ensure score is within bounds
        return max(0.0, min(1.0, score))
    
    def _enhance_with_cross_validation(
        self, 
        best_result: OCRResult, 
        all_results: List[OCRResult]
    ) -> Tuple[str, float]:
        """Enhance the best result using cross-validation with other results."""
        if len(all_results) <= 1:
            return best_result.text, best_result.confidence
        
        # Get other results for comparison
        other_results = [r for r in all_results if r.engine != best_result.engine]
        
        if not other_results:
            return best_result.text, best_result.confidence
        
        enhanced_text = best_result.text
        enhanced_confidence = best_result.confidence
        
        # Cross-validate with other results
        similarities = []
        for other_result in other_results:
            similarity = self._calculate_text_similarity(
                best_result.text, other_result.text
            )
            similarities.append(similarity)
        
        if similarities:
            avg_similarity = statistics.mean(similarities)
            max_similarity = max(similarities)
            
            # Boost confidence if results are similar
            if avg_similarity > self.similarity_threshold:
                confidence_boost = (avg_similarity - self.similarity_threshold) * 0.5
                enhanced_confidence = min(1.0, enhanced_confidence + confidence_boost)
            
            # If similarity is very high, try to merge/correct text
            if max_similarity > 0.9:
                enhanced_text = self._merge_similar_texts(best_result, other_results)
        
        return enhanced_text, enhanced_confidence
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        
        # Normalize texts for comparison
        norm_text1 = self._normalize_text_for_comparison(text1)
        norm_text2 = self._normalize_text_for_comparison(text2)
        
        # Use sequence matcher for similarity
        matcher = SequenceMatcher(None, norm_text1, norm_text2)
        return matcher.ratio()
    
    def _normalize_text_for_comparison(self, text: str) -> str:
        """Normalize text for similarity comparison."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation for core content comparison
        text = re.sub(r'[^\w\s]', '', text)
        
        return text.strip()
    
    def _merge_similar_texts(
        self, 
        best_result: OCRResult, 
        other_results: List[OCRResult]
    ) -> str:
        """Merge very similar texts to correct potential OCR errors."""
        # Find the most similar result
        best_similarity = 0
        most_similar = None
        
        for result in other_results:
            similarity = self._calculate_text_similarity(
                best_result.text, result.text
            )
            if similarity > best_similarity:
                best_similarity = similarity
                most_similar = result
        
        if not most_similar or best_similarity < 0.9:
            return best_result.text
        
        # Simple merge strategy: use the longer text if very similar
        text1 = best_result.text.strip()
        text2 = most_similar.text.strip()
        
        if len(text2) > len(text1) * 1.1:  # Other result is significantly longer
            return text2
        else:
            return text1
    
    def analyze_ocr_quality(self, raw_content: RawContent) -> Dict[str, Any]:
        """Analyze the quality of OCR results for debugging and optimization."""
        analysis = {
            'total_engines': len(raw_content.ocr_results),
            'successful_engines': sum(1 for r in raw_content.ocr_results if r.success),
            'engine_results': {},
            'best_engine': raw_content.best_engine,
            'merged_confidence': raw_content.merged_confidence,
            'text_length': len(raw_content.merged_text),
            'processing_times': {},
            'recommendations': []
        }
        
        # Analyze each engine result
        for result in raw_content.ocr_results:
            engine_name = result.engine.value
            analysis['engine_results'][engine_name] = {
                'success': result.success,
                'confidence': result.confidence,
                'text_length': len(result.text),
                'processing_time': result.processing_time,
                'error': result.error_message
            }
            analysis['processing_times'][engine_name] = result.processing_time
        
        # Generate recommendations
        successful_results = [r for r in raw_content.ocr_results if r.success]
        
        if not successful_results:
            analysis['recommendations'].append("All OCR engines failed - check PDF quality")
        elif len(successful_results) == 1:
            analysis['recommendations'].append("Only one engine succeeded - consider PDF preprocessing")
        elif raw_content.merged_confidence < 0.5:
            analysis['recommendations'].append("Low confidence results - manual review recommended")
        
        # Check for processing time issues
        avg_time = statistics.mean(analysis['processing_times'].values()) if analysis['processing_times'] else 0
        for engine, time in analysis['processing_times'].items():
            if time > avg_time * 3:
                analysis['recommendations'].append(f"{engine} is slow - check configuration")
        
        return analysis