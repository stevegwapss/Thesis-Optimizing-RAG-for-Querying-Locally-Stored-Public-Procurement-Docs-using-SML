"""
Main PDF Processing Orchestrator

This module coordinates all stages of the PDF processing pipeline:
1. Metadata extraction
2. OCR processing and merging
3. Sentence chunking
4. Section tagging
5. Section-aware chunking
6. Embedding generation
7. Vector database storage
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
import traceback

from .models import (
    DocumentMetadata, ProcessingResult, ProcessingStats, ConfigParameters,
    ContentType, OCREngine
)
from .metadata_extractor import MetadataExtractor
from .ocr_engines import OCREngineManager
from .ocr_merger import OCRMerger
from .sentence_chunker import SentenceChunker
from .section_tagger import SectionTagger
from .section_aware_chunker import SectionAwareChunker
from .embedding_generator import EmbeddingGenerator
from .vector_store import VectorStoreManager


logger = logging.getLogger(__name__)


class PDFProcessor:
    """Main orchestrator for the PDF processing pipeline."""
    
    def __init__(
        self, 
        config: Optional[ConfigParameters] = None,
        **kwargs
    ):
        """
        Initialize PDF processor with configuration.
        
        Args:
            config: Configuration parameters
            **kwargs: Override specific config values
        """
        # Set up configuration
        if config is None:
            config = ConfigParameters()
        
        # Apply any overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        self.config = config
        
        # Initialize components
        self.metadata_extractor = MetadataExtractor()
        self.ocr_manager = OCREngineManager(config.model_dump())
        self.ocr_merger = OCRMerger(config.model_dump())
        self.sentence_chunker = SentenceChunker(config.model_dump())
        self.section_tagger = SectionTagger(config.model_dump())
        self.section_aware_chunker = SectionAwareChunker(config)
        self.embedding_generator = EmbeddingGenerator(config)
        self.vector_store_manager = VectorStoreManager(config)
        
        # Processing statistics
        self.processing_stats = ProcessingStats(
            total_pages=0,
            total_chunks=0,
            total_sentences=0,
            total_tables=0,
            avg_confidence=0.0,
            min_confidence=1.0,
            max_confidence=0.0,
            total_processing_time=0.0
        )
        
        logger.info("Initialized PDF processor with configuration")
    
    async def process_document(
        self, 
        pdf_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """
        Process a complete PDF document through the full pipeline.
        
        Args:
            pdf_path: Path to the PDF file
            metadata: Additional metadata to include
            
        Returns:
            ProcessingResult with all processing outputs
        """
        start_time = time.time()
        stage_times = {}
        
        try:
            logger.info(f"Starting processing of document: {pdf_path}")
            
            # Validate file exists
            if not Path(pdf_path).exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            # Stage 1: Extract metadata
            stage_start = time.time()
            document_metadata = await self._stage_1_extract_metadata(pdf_path, metadata)
            stage_times['metadata_extraction'] = time.time() - stage_start
            
            # Initialize vector store
            stage_start = time.time()
            await self.vector_store_manager.initialize()
            stage_times['vector_store_init'] = time.time() - stage_start
            
            # Stage 2: Process all pages with OCR
            stage_start = time.time()
            raw_contents = await self._stage_2_ocr_processing(pdf_path, document_metadata)
            stage_times['ocr_processing'] = time.time() - stage_start
            
            # Stage 3: Convert to sentence chunks
            stage_start = time.time()
            all_sentences = await self._stage_3_sentence_chunking(raw_contents, document_metadata)
            stage_times['sentence_chunking'] = time.time() - stage_start
            
            # Stage 4: Tag sentences with sections
            stage_start = time.time()
            tagged_sentences = await self._stage_4_section_tagging(all_sentences, raw_contents, document_metadata)
            stage_times['section_tagging'] = time.time() - stage_start
            
            # Stage 5: Create contextual chunks
            stage_start = time.time()
            contextual_chunks = await self._stage_5_contextual_chunking(tagged_sentences)
            stage_times['contextual_chunking'] = time.time() - stage_start
            
            # Stage 6: Generate embeddings
            stage_start = time.time()
            embedding_results = await self._stage_6_generate_embeddings(contextual_chunks)
            stage_times['embedding_generation'] = time.time() - stage_start
            
            # Stage 7: Store in vector database
            stage_start = time.time()
            vector_ids = await self._stage_7_vector_storage(embedding_results, document_metadata)
            stage_times['vector_storage'] = time.time() - stage_start
            
            # Calculate final statistics
            total_time = time.time() - start_time
            self.processing_stats.total_processing_time = total_time
            self.processing_stats.stage_times = stage_times
            
            # Create result
            result = ProcessingResult(
                document_metadata=document_metadata,
                processing_stats=self.processing_stats,
                chunks=contextual_chunks,
                embeddings=embedding_results,
                vector_ids=vector_ids,
                collection_name=self.config.collection_name,
                processing_completed_at=time.time(),
                success=True
            )
            
            logger.info(f"Successfully processed document in {total_time:.2f}s: "
                       f"{result.chunk_count} chunks, {result.table_count} tables")
            
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            error_msg = f"Error processing document {pdf_path}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            # Create error result
            result = ProcessingResult(
                document_metadata=document_metadata or DocumentMetadata(
                    pdf_type="unknown",
                    page_count=0,
                    file_size=0,
                    file_path=pdf_path
                ),
                processing_stats=ProcessingStats(
                    total_pages=0,
                    total_chunks=0,
                    total_sentences=0,
                    total_tables=0,
                    avg_confidence=0.0,
                    min_confidence=0.0,
                    max_confidence=0.0,
                    total_processing_time=total_time,
                    stage_times=stage_times
                ),
                chunks=[],
                embeddings=[],
                vector_ids=[],
                collection_name=self.config.collection_name,
                success=False,
                error_messages=[error_msg]
            )
            
            return result
    
    async def _stage_1_extract_metadata(
        self, 
        pdf_path: str,
        custom_metadata: Optional[Dict[str, Any]]
    ) -> DocumentMetadata:
        """Stage 1: Extract document metadata."""
        logger.debug("Stage 1: Extracting metadata")
        
        metadata = await self.metadata_extractor.extract_metadata(pdf_path, custom_metadata)
        
        # Update processing stats
        self.processing_stats.total_pages = metadata.page_count
        
        return metadata
    
    async def _stage_2_ocr_processing(
        self, 
        pdf_path: str,
        metadata: DocumentMetadata
    ) -> List[Any]:  # RawContent list
        """Stage 2: OCR processing for all pages."""
        logger.debug(f"Stage 2: OCR processing {metadata.page_count} pages")
        
        raw_contents = []
        confidences = []
        
        # Process pages in batches for memory efficiency
        max_concurrent_pages = min(self.config.max_workers, 4)
        
        for start_page in range(0, metadata.page_count, max_concurrent_pages):
            end_page = min(start_page + max_concurrent_pages, metadata.page_count)
            
            # Create tasks for this batch
            tasks = []
            for page_num in range(start_page, end_page):
                task = self.ocr_manager.extract_page_content(
                    pdf_path, page_num, metadata, self.config.ocr_engines
                )
                tasks.append(task)
            
            # Process batch
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Merge OCR results and collect
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"OCR processing failed for page: {result}")
                    continue
                
                # Merge OCR results
                merged_content = self.ocr_merger.merge_ocr_results(result, metadata.pdf_type)
                raw_contents.append(merged_content)
                
                if merged_content.merged_confidence > 0:
                    confidences.append(merged_content.merged_confidence)
        
        # Update statistics
        if confidences:
            self.processing_stats.avg_confidence = sum(confidences) / len(confidences)
            self.processing_stats.min_confidence = min(confidences)
            self.processing_stats.max_confidence = max(confidences)
        
        # Count OCR engine usage
        for content in raw_contents:
            for ocr_result in content.ocr_results:
                if ocr_result.engine not in self.processing_stats.ocr_engine_usage:
                    self.processing_stats.ocr_engine_usage[ocr_result.engine] = 0
                self.processing_stats.ocr_engine_usage[ocr_result.engine] += 1
        
        return raw_contents
    
    async def _stage_3_sentence_chunking(
        self, 
        raw_contents: List[Any],
        metadata: DocumentMetadata
    ) -> List[Any]:  # SentenceChunk list
        """Stage 3: Convert raw content to sentence chunks."""
        logger.debug("Stage 3: Sentence chunking")
        
        all_sentences = []
        
        # Process each page's content
        for content in raw_contents:
            try:
                sentences = await self.sentence_chunker.chunk_into_sentences(content, metadata)
                all_sentences.extend(sentences)
                
                # Process table content separately
                if content.tables:
                    table_sentences = await self.sentence_chunker.chunk_table_content(
                        content.tables, content
                    )
                    all_sentences.extend(table_sentences)
                    
            except Exception as e:
                logger.error(f"Error chunking sentences for page {content.page_number}: {e}")
                continue
        
        # Update statistics
        self.processing_stats.total_sentences = len(all_sentences)
        
        return all_sentences
    
    async def _stage_4_section_tagging(
        self, 
        sentences: List[Any],
        raw_contents: List[Any],
        metadata: DocumentMetadata
    ) -> List[Any]:  # TaggedSentence list
        """Stage 4: Tag sentences with section information."""
        logger.debug("Stage 4: Section tagging")
        
        # Collect all tables from raw contents
        all_tables = []
        for content in raw_contents:
            all_tables.extend(content.tables)
        
        # Tag sentences
        tagged_sentences = await self.section_tagger.tag_sentences(
            sentences, all_tables, metadata
        )
        
        # Update statistics
        content_type_counts = {}
        for sentence in tagged_sentences:
            content_type = sentence.content_type
            if content_type not in content_type_counts:
                content_type_counts[content_type] = 0
            content_type_counts[content_type] += 1
        
        self.processing_stats.content_type_counts = content_type_counts
        self.processing_stats.total_tables = content_type_counts.get(ContentType.TABLE, 0)
        
        return tagged_sentences
    
    async def _stage_5_contextual_chunking(
        self, 
        tagged_sentences: List[Any]
    ) -> List[Any]:  # ContextualChunk list
        """Stage 5: Create section-aware contextual chunks."""
        logger.debug("Stage 5: Contextual chunking")
        
        chunks = await self.section_aware_chunker.create_contextual_chunks(tagged_sentences)
        
        # Update statistics
        self.processing_stats.total_chunks = len(chunks)
        
        return chunks
    
    async def _stage_6_generate_embeddings(
        self, 
        chunks: List[Any]
    ) -> List[Any]:  # EmbeddingResult list
        """Stage 6: Generate embeddings for chunks."""
        logger.debug("Stage 6: Generating embeddings")
        
        embedding_results = await self.embedding_generator.generate_embeddings(chunks)
        
        return embedding_results
    
    async def _stage_7_vector_storage(
        self, 
        embedding_results: List[Any],
        metadata: DocumentMetadata
    ) -> List[str]:
        """Stage 7: Store embeddings in vector database."""
        logger.debug("Stage 7: Vector storage")
        
        vector_ids = await self.vector_store_manager.store_document_embeddings(
            embedding_results, metadata
        )
        
        return vector_ids
    
    async def process_multiple_documents(
        self, 
        pdf_paths: List[str],
        batch_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[ProcessingResult]:
        """
        Process multiple PDF documents.
        
        Args:
            pdf_paths: List of PDF file paths
            batch_metadata: List of metadata dicts (optional)
            
        Returns:
            List of ProcessingResult objects
        """
        logger.info(f"Processing {len(pdf_paths)} documents")
        
        if batch_metadata and len(batch_metadata) != len(pdf_paths):
            raise ValueError("Metadata list length must match PDF paths length")
        
        results = []
        
        # Process documents with limited concurrency
        max_concurrent = min(self.config.max_workers, 2)  # Limit for memory
        
        for i in range(0, len(pdf_paths), max_concurrent):
            batch_paths = pdf_paths[i:i + max_concurrent]
            batch_meta = batch_metadata[i:i + max_concurrent] if batch_metadata else [None] * len(batch_paths)
            
            # Create tasks for this batch
            tasks = []
            for pdf_path, meta in zip(batch_paths, batch_meta):
                task = self.process_document(pdf_path, meta)
                tasks.append(task)
            
            # Process batch
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect results
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Document processing failed: {result}")
                    # Create error result
                    error_result = ProcessingResult(
                        document_metadata=DocumentMetadata(
                            pdf_type="unknown",
                            page_count=0,
                            file_size=0,
                            file_path="unknown"
                        ),
                        processing_stats=ProcessingStats(
                            total_pages=0,
                            total_chunks=0,
                            total_sentences=0,
                            total_tables=0,
                            avg_confidence=0.0,
                            min_confidence=0.0,
                            max_confidence=0.0,
                            total_processing_time=0.0
                        ),
                        chunks=[],
                        embeddings=[],
                        vector_ids=[],
                        collection_name=self.config.collection_name,
                        success=False,
                        error_messages=[str(result)]
                    )
                    results.append(error_result)
                else:
                    results.append(result)
        
        successful = sum(1 for r in results if r.success)
        logger.info(f"Completed processing: {successful}/{len(results)} documents successful")
        
        return results
    
    async def search_documents(
        self, 
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search processed documents using text query.
        
        Args:
            query: Search query text
            limit: Maximum number of results
            filters: Optional filters (doc_type, department, etc.)
            
        Returns:
            List of search results
        """
        try:
            # Generate query embedding
            query_embedding = await self.embedding_generator.text_embedder.generate_embedding(query)
            
            # Search vector store
            results = await self.vector_store_manager.search(
                query_vector=query_embedding,
                limit=limit,
                filters=filters
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete all data for a specific document."""
        try:
            return await self.vector_store_manager.delete_document(document_id)
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        try:
            vector_stats = await self.vector_store_manager.get_stats()
            
            system_stats = {
                'vector_store': vector_stats,
                'configuration': {
                    'ocr_engines': [e.value for e in self.config.ocr_engines],
                    'embedding_model': self.config.embedding_model,
                    'target_chunk_size': self.config.target_chunk_size,
                    'vector_store_type': self.config.vector_store_type
                },
                'ocr_engines_available': list(self.ocr_manager.available_engines.keys()),
                'last_processing_stats': self.processing_stats.dict() if self.processing_stats else None
            }
            
            return system_stats
            
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {'error': str(e)}