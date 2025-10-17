"""
PDF Chunking Pipeline - Main Orchestrator

Coordinates all 7 stages:
1. File System Handling (OS + pathlib)
2. Triple OCR Extraction (Parallel)
3. spaCy Text Processing
4. spaCy PhraseMatcher
5. Structured Extraction (regex + PDFPlumber)
6. Dual Embedding Generation
7. ChromaDB Storage
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import json
import yaml
from datetime import datetime

# Import pipeline components
from .file_handler import FileSystemHandler
from .ocr_engines import MultiOCRProcessor, OCRResult
from .spacy_processor import SpaCyProcessor
from .phrase_matcher import ProcurementPhraseMatcher
from .structured_extractor import StructuredExtractor
from .embedding_generator import DualEmbedder
from .vector_store import ChromaVectorStore, create_text_chunk, create_table_chunk

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Result from document processing."""
    success: bool
    document_path: str
    processing_time: float
    chunks_created: int
    tables_extracted: int
    embeddings_generated: int
    metadata: Dict[str, Any]
    errors: List[str]

@dataclass
class PipelineStats:
    """Statistics about pipeline processing."""
    total_documents: int
    successful_documents: int
    failed_documents: int
    total_chunks: int
    total_tables: int
    total_embeddings: int
    total_processing_time: float
    avg_processing_time: float

class PDFChunkingPipeline:
    """Main orchestrator for the PDF chunking pipeline."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        self.stats = PipelineStats(0, 0, 0, 0, 0, 0, 0.0, 0.0)
        
        # Initialize components
        self._initialize_components()
        
        logger.info("PDF Chunking Pipeline initialized")
    
    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load configuration from file or use defaults."""
        default_config = {
            'file_handling': {
                'temp_dir': './temp',
                'output_dir': './output',
                'cleanup_on_complete': True
            },
            'ocr': {
                'engines': ['tesseract', 'pypdf2', 'pdfplumber'],
                'tesseract_config': '--oem 3 --psm 6',
                'confidence_threshold': 0.7,
                'batch_size': 5
            },
            'spacy': {
                'model': 'en_core_web_sm',
                'batch_size': 100,
                'n_process': 4
            },
            'embeddings': {
                'text_model': 'all-MiniLM-L6-v2',
                'table_model': 'google/tapas-base',
                'batch_size': 32,
                'max_text_length': 512
            },
            'chunking': {
                'target_size': 600,
                'overlap': 100,
                'preserve_tables': True,
                'preserve_headers': True
            },
            'vector_store': {
                'provider': 'chromadb',
                'persist_directory': './chroma_db',
                'collection_name': 'procurement_docs'
            }
        }
        
        if config_path and config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    if config_path.suffix.lower() == '.yaml':
                        user_config = yaml.safe_load(f)
                    else:
                        user_config = json.load(f)
                
                # Merge with defaults
                default_config.update(user_config)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.warning(f"Could not load config from {config_path}: {e}")
        
        return default_config
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        try:
            # Stage 1: File System Handler
            self.file_handler = FileSystemHandler()
            self.file_handler.setup(self.config['file_handling'].get('output_dir'))
            
            # Stage 2: OCR Processor
            self.ocr_processor = MultiOCRProcessor(self.config['ocr'])
            
            # Stage 3: spaCy Processor
            spacy_model = self.config['spacy'].get('model', 'en_core_web_sm')
            self.spacy_processor = SpaCyProcessor(spacy_model)
            
            # Stage 4: Phrase Matcher
            if self.spacy_processor.is_available():
                self.phrase_matcher = ProcurementPhraseMatcher(self.spacy_processor.nlp)
            else:
                logger.warning("spaCy not available, phrase matching disabled")
                self.phrase_matcher = None
            
            # Stage 5: Structured Extractor
            self.extractor = StructuredExtractor()
            
            # Stage 6: Dual Embedder
            self.embedder = DualEmbedder(self.config['embeddings'])
            
            # Stage 7: Vector Store
            vector_config = self.config['vector_store']
            self.vector_store = ChromaVectorStore(
                persist_directory=vector_config.get('persist_directory', './chroma_db'),
                config=vector_config
            )
            
            # Create default collection
            collection_name = vector_config.get('collection_name', 'procurement_docs')
            self.vector_store.create_collection(collection_name)
            
            logger.info("All pipeline components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing pipeline components: {e}")
            raise
    
    async def process_document(self, pdf_path: Path) -> ProcessingResult:
        """Process a single PDF document through the entire pipeline."""
        start_time = time.time()
        errors = []
        
        try:
            pdf_path = Path(pdf_path)
            logger.info(f"Starting processing of {pdf_path}")
            
            # Validate input
            if not self.file_handler.validate_pdf(pdf_path):
                error_msg = f"PDF validation failed for {pdf_path}"
                logger.error(error_msg)
                return ProcessingResult(
                    success=False,
                    document_path=str(pdf_path),
                    processing_time=time.time() - start_time,
                    chunks_created=0,
                    tables_extracted=0,
                    embeddings_generated=0,
                    metadata={},
                    errors=[error_msg]
                )
            
            # Stage 1: File handling (already done in validation)
            logger.info("✅ Stage 1: File handling completed")
            
            # Stage 2: Triple OCR extraction
            logger.info("🔄 Stage 2: Running triple OCR extraction...")
            ocr_results = await self._stage2_ocr_extraction(pdf_path)
            if not ocr_results:
                error_msg = "No OCR results obtained"
                errors.append(error_msg)
                logger.error(error_msg)
            else:
                logger.info(f"✅ Stage 2: OCR completed - {len(ocr_results)} pages processed")
            
            # Stage 3: spaCy processing
            logger.info("🔄 Stage 3: spaCy text processing...")
            processed_text_data = await self._stage3_spacy_processing(ocr_results)
            logger.info("✅ Stage 3: spaCy processing completed")
            
            # Stage 4: Phrase matching
            logger.info("🔄 Stage 4: Phrase pattern matching...")
            phrase_matches = await self._stage4_phrase_matching(processed_text_data)
            logger.info(f"✅ Stage 4: Found {len(phrase_matches)} phrase matches")
            
            # Stage 5: Structured extraction
            logger.info("🔄 Stage 5: Structured data extraction...")
            structured_data = await self._stage5_structured_extraction(pdf_path, processed_text_data)
            tables_count = len(structured_data.get('tables', []))
            logger.info(f"✅ Stage 5: Extracted {tables_count} tables and structured data")
            
            # Stage 6: Create chunks
            logger.info("🔄 Stage 6: Creating intelligent chunks...")
            chunks = await self._stage6_create_chunks(
                pdf_path, processed_text_data, phrase_matches, structured_data
            )
            logger.info(f"✅ Stage 6: Created {len(chunks)} chunks")
            
            # Stage 7: Generate embeddings
            logger.info("🔄 Stage 7: Generating dual embeddings...")
            embeddings_count = await self._stage7_generate_embeddings(chunks)
            logger.info(f"✅ Stage 7: Generated {embeddings_count} embeddings")
            
            # Stage 8: Store in vector database
            logger.info("🔄 Stage 8: Storing in ChromaDB...")
            storage_success = await self._stage8_vector_storage(chunks)
            if storage_success:
                logger.info("✅ Stage 8: Vector storage completed")
            else:
                errors.append("Vector storage failed")
            
            # Update statistics
            self.stats.total_documents += 1
            if not errors:
                self.stats.successful_documents += 1
            else:
                self.stats.failed_documents += 1
            
            self.stats.total_chunks += len(chunks)
            self.stats.total_tables += tables_count
            self.stats.total_embeddings += embeddings_count
            
            processing_time = time.time() - start_time
            self.stats.total_processing_time += processing_time
            self.stats.avg_processing_time = self.stats.total_processing_time / self.stats.total_documents
            
            # Create metadata
            metadata = {
                'file_info': self.file_handler.get_file_info(pdf_path),
                'ocr_engines_used': [engine for engine in ['tesseract', 'pypdf2', 'pdfplumber'] 
                                   if self.ocr_processor.available_engines.get(engine)],
                'phrase_matches_count': len(phrase_matches),
                'spacy_available': self.spacy_processor.is_available(),
                'embedding_models': self.embedder.get_embedding_dimensions(),
                'processing_timestamp': datetime.now().isoformat()
            }
            
            result = ProcessingResult(
                success=len(errors) == 0,
                document_path=str(pdf_path),
                processing_time=processing_time,
                chunks_created=len(chunks),
                tables_extracted=tables_count,
                embeddings_generated=embeddings_count,
                metadata=metadata,
                errors=errors
            )
            
            logger.info(f"🎉 Document processing completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Pipeline error: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            
            return ProcessingResult(
                success=False,
                document_path=str(pdf_path),
                processing_time=processing_time,
                chunks_created=0,
                tables_extracted=0,
                embeddings_generated=0,
                metadata={},
                errors=errors
            )
    
    async def _stage2_ocr_extraction(self, pdf_path: Path) -> Dict[int, List[OCRResult]]:
        """Stage 2: Triple OCR extraction."""
        try:
            results = await self.ocr_processor.process_all_pages(pdf_path)
            return results
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return {}
    
    async def _stage3_spacy_processing(self, ocr_results: Dict[int, List[OCRResult]]) -> Dict:
        """Stage 3: spaCy text processing."""
        try:
            all_text = ""
            merged_results = []
            
            for page_num, page_results in ocr_results.items():
                if page_results:
                    # Merge OCR results for this page
                    merged_result = self.ocr_processor.merge_results(page_results)
                    if merged_result and merged_result.text:
                        all_text += f"\n{merged_result.text}"
                        merged_results.append({
                            'page': page_num,
                            'text': merged_result.text,
                            'confidence': merged_result.confidence,
                            'metadata': merged_result.metadata
                        })
            
            # Process all text with spaCy
            if all_text.strip():
                processed = self.spacy_processor.process_text(all_text)
                processed['merged_ocr_results'] = merged_results
                return processed
            else:
                return {'merged_ocr_results': merged_results}
                
        except Exception as e:
            logger.error(f"spaCy processing failed: {e}")
            return {}
    
    async def _stage4_phrase_matching(self, processed_text_data: Dict) -> List[Dict]:
        """Stage 4: Phrase pattern matching."""
        try:
            if not self.phrase_matcher or not processed_text_data.get('doc'):
                return []
            
            matches = self.phrase_matcher.find_matches(processed_text_data['doc'])
            
            return [
                {
                    'text': match.text,
                    'label': match.label,
                    'content_type': match.content_type.value,
                    'start_char': match.start_char,
                    'end_char': match.end_char,
                    'confidence': match.confidence,
                    'context': match.context
                } for match in matches
            ]
            
        except Exception as e:
            logger.error(f"Phrase matching failed: {e}")
            return []
    
    async def _stage5_structured_extraction(self, pdf_path: Path, processed_text_data: Dict) -> Dict:
        """Stage 5: Structured data extraction."""
        try:
            # Extract text for regex patterns
            text = ""
            for result in processed_text_data.get('merged_ocr_results', []):
                text += " " + result.get('text', '')
            
            # Extract tables using PDFPlumber
            tables = self.extractor.extract_tables_with_pdfplumber(pdf_path)
            
            # Build comprehensive structured document
            structured_doc = self.extractor.build_structured_document(text, tables)
            
            return structured_doc
            
        except Exception as e:
            logger.error(f"Structured extraction failed: {e}")
            return {'tables': []}
    
    async def _stage6_create_chunks(
        self,
        pdf_path: Path,
        processed_text_data: Dict,
        phrase_matches: List[Dict],
        structured_data: Dict
    ) -> List:
        """Stage 6: Create intelligent chunks."""
        try:
            chunks = []
            source_document = pdf_path.name
            
            # Create text chunks from processed sentences
            sentences = processed_text_data.get('sentences', [])
            for i, sentence in enumerate(sentences):
                if isinstance(sentence, dict):
                    text = sentence.get('text', '')
                else:
                    text = str(sentence)
                
                if text.strip():
                    # Determine which page this sentence belongs to
                    page_num = self._estimate_page_number(i, len(sentences), processed_text_data)
                    
                    # Create metadata with phrase matches for this sentence
                    metadata = {
                        'sentence_index': i,
                        'sentence_length': len(text),
                        'has_entities': len(sentence.get('entities', [])) > 0 if isinstance(sentence, dict) else False
                    }
                    
                    # Add relevant phrase matches
                    relevant_matches = [
                        match for match in phrase_matches
                        if match['text'].lower() in text.lower()
                    ]
                    if relevant_matches:
                        metadata['phrase_matches'] = relevant_matches
                        metadata['content_types'] = list(set(match['content_type'] for match in relevant_matches))
                    
                    chunk = create_text_chunk(
                        text=text,
                        source_document=source_document,
                        page_number=page_num,
                        position=i,
                        metadata=metadata,
                        confidence_score=sentence.get('confidence', 1.0) if isinstance(sentence, dict) else 1.0
                    )
                    chunks.append(chunk)
            
            # Create table chunks
            tables = structured_data.get('tables', [])
            for table in tables:
                table_metadata = {
                    'table_type': table.get('metadata', {}).get('table_type', 'data'),
                    'has_financial_data': table.get('metadata', {}).get('has_financial_data', False),
                    'has_dates': table.get('metadata', {}).get('has_dates', False)
                }
                
                chunk = create_table_chunk(
                    table_data=table,
                    source_document=source_document,
                    page_number=table.get('metadata', {}).get('page_number', 0),
                    position=len(chunks),  # Position after text chunks
                    metadata=table_metadata,
                    confidence_score=0.9  # High confidence for structured tables
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Chunk creation failed: {e}")
            return []
    
    async def _stage7_generate_embeddings(self, chunks: List) -> int:
        """Stage 7: Generate dual embeddings."""
        try:
            embeddings_count = 0
            
            # Separate text and table chunks
            text_chunks = [chunk for chunk in chunks if chunk.chunk_type == 'text']
            table_chunks = [chunk for chunk in chunks if chunk.chunk_type == 'table']
            
            # Generate text embeddings in batches
            if text_chunks:
                batch_size = self.config['embeddings'].get('batch_size', 32)
                texts = [chunk.text for chunk in text_chunks]
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_chunks = text_chunks[i:i + batch_size]
                    
                    try:
                        embedding_results = self.embedder.batch_embed_texts(batch_texts, len(batch_texts))
                        
                        for chunk, result in zip(batch_chunks, embedding_results):
                            chunk.embedding = result.embedding
                            chunk.metadata['embedding_model'] = result.model_name
                            chunk.metadata['embedding_time'] = result.processing_time
                            embeddings_count += 1
                            
                    except Exception as e:
                        logger.warning(f"Text embedding batch failed: {e}")
            
            # Generate table embeddings individually
            for chunk in table_chunks:
                try:
                    # Reconstruct table data from chunk metadata
                    table_data = {
                        'headers': chunk.metadata.get('headers', []),
                        'rows': chunk.metadata.get('rows', []),
                        'metadata': chunk.metadata
                    }
                    
                    result = self.embedder.embed_table(table_data)
                    chunk.embedding = result.embedding
                    chunk.metadata['embedding_model'] = result.model_name
                    chunk.metadata['embedding_time'] = result.processing_time
                    embeddings_count += 1
                    
                except Exception as e:
                    logger.warning(f"Table embedding failed for chunk {chunk.chunk_id}: {e}")
            
            return embeddings_count
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return 0
    
    async def _stage8_vector_storage(self, chunks: List) -> bool:
        """Stage 8: Store in ChromaDB."""
        try:
            collection_name = self.config['vector_store'].get('collection_name', 'procurement_docs')
            
            # Filter chunks with embeddings
            chunks_with_embeddings = [chunk for chunk in chunks if chunk.embedding is not None]
            
            if not chunks_with_embeddings:
                logger.warning("No chunks with embeddings to store")
                return False
            
            # Store in vector database
            success = self.vector_store.upsert_chunks(collection_name, chunks_with_embeddings)
            
            return success
            
        except Exception as e:
            logger.error(f"Vector storage failed: {e}")
            return False
    
    def _estimate_page_number(self, sentence_index: int, total_sentences: int, processed_data: Dict) -> int:
        """Estimate page number for a sentence based on its position."""
        # Simple estimation based on OCR results
        merged_results = processed_data.get('merged_ocr_results', [])
        if not merged_results:
            return 0
        
        # Estimate based on position ratio
        ratio = sentence_index / max(total_sentences, 1)
        estimated_page = int(ratio * len(merged_results))
        
        return min(estimated_page, len(merged_results) - 1)
    
    async def process_batch(self, pdf_paths: List[Path], max_concurrent: int = 3) -> List[ProcessingResult]:
        """Process multiple PDFs concurrently."""
        logger.info(f"Starting batch processing of {len(pdf_paths)} documents")
        
        # Create semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(pdf_path):
            async with semaphore:
                return await self.process_document(pdf_path)
        
        # Process all documents
        tasks = [process_with_semaphore(pdf_path) for pdf_path in pdf_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Document {pdf_paths[i]} failed: {result}")
                processed_results.append(ProcessingResult(
                    success=False,
                    document_path=str(pdf_paths[i]),
                    processing_time=0.0,
                    chunks_created=0,
                    tables_extracted=0,
                    embeddings_generated=0,
                    metadata={},
                    errors=[str(result)]
                ))
            else:
                processed_results.append(result)
        
        logger.info(f"Batch processing completed: {len(processed_results)} results")
        return processed_results
    
    def search(
        self,
        query: str,
        filters: Optional[Dict] = None,
        top_k: int = 5,
        collection_name: Optional[str] = None
    ) -> List[Dict]:
        """Search the processed documents."""
        try:
            collection = collection_name or self.config['vector_store'].get('collection_name', 'procurement_docs')
            
            results = self.vector_store.search(
                collection_name=collection,
                query_text=query,
                filters=filters,
                top_k=top_k
            )
            
            return [
                {
                    'text': result.text,
                    'score': result.score,
                    'metadata': result.metadata,
                    'source': result.metadata.get('source_document', 'unknown'),
                    'page': result.metadata.get('page_number', 0),
                    'chunk_type': result.metadata.get('chunk_type', 'unknown')
                } for result in results
            ]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """Get pipeline processing statistics."""
        return asdict(self.stats)
    
    def save_stats(self, output_path: Path):
        """Save statistics to file."""
        try:
            stats_data = {
                'pipeline_stats': asdict(self.stats),
                'component_availability': {
                    'file_handler': True,
                    'ocr_engines': list(self.ocr_processor.available_engines.keys()),
                    'spacy_processor': self.spacy_processor.is_available(),
                    'phrase_matcher': self.phrase_matcher is not None,
                    'structured_extractor': True,
                    'dual_embedder': self.embedder.is_available(),
                    'vector_store': self.vector_store.is_available()
                },
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(output_path, 'w') as f:
                json.dump(stats_data, f, indent=2)
            
            logger.info(f"Stats saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save stats: {e}")
    
    def cleanup(self):
        """Cleanup pipeline resources."""
        try:
            if self.config['file_handling'].get('cleanup_on_complete', True):
                self.file_handler.cleanup_temp_files()
            
            self.vector_store.cleanup()
            logger.info("Pipeline cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during cleanup