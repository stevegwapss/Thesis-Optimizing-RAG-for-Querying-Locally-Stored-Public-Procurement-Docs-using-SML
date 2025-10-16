"""
Example usage of the Advanced PDF Chunking Pipeline

This script demonstrates how to use the PDF processing system
to process procurement documents and store them in a vector database.
"""

import asyncio
import logging
import os
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.pdf_processor import PDFProcessor
from src.models import ConfigParameters, OCREngine


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def example_single_document():
    """Example: Process a single PDF document."""
    
    # Initialize processor with custom configuration
    processor = PDFProcessor(
        ocr_engines=[OCREngine.TESSERACT, OCREngine.PYPDF2, OCREngine.PDFPLUMBER],
        embedding_model='text-embedding-3-large',
        vector_store_type='qdrant',
        chunk_size=600,
        chunk_overlap=100
    )
    
    # Example PDF path (update with your actual file)
    pdf_path = "model-training/my_pdfs/01. Annual Procurement Plan - Non CSE FY 2024.pdf"
    
    if not Path(pdf_path).exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return None
    
    # Process document with metadata
    metadata = {
        'doc_type': 'procurement_plan',
        'fiscal_year': 2024,
        'department': 'DOH'
    }
    
    logger.info(f"Processing document: {pdf_path}")
    result = await processor.process_document(pdf_path, metadata)
    
    if result.success:
        logger.info(f"✅ Successfully processed document!")
        logger.info(f"📄 Created {result.chunk_count} chunks")
        logger.info(f"📊 Found {result.table_count} tables")
        logger.info(f"🎯 Average confidence: {result.avg_confidence:.3f}")
        logger.info(f"🗂️ Vector IDs: {len(result.vector_ids)} embeddings stored")
    else:
        logger.error(f"❌ Processing failed: {result.error_messages}")
    
    return result


async def example_multiple_documents():
    """Example: Process multiple PDF documents in batch."""
    
    processor = PDFProcessor()
    
    # Find all PDF files in the directory
    pdf_dir = Path("model-training/my_pdfs")
    pdf_files = list(pdf_dir.glob("*.pdf"))[:3]  # Process first 3 files
    
    if not pdf_files:
        logger.error(f"No PDF files found in {pdf_dir}")
        return []
    
    logger.info(f"Processing {len(pdf_files)} documents")
    
    # Process batch with metadata
    batch_metadata = []
    for pdf_file in pdf_files:
        filename = pdf_file.stem
        
        # Extract fiscal year from filename
        fiscal_year = None
        if "FY 2024" in filename:
            fiscal_year = 2024
        elif "FY 2025" in filename:
            fiscal_year = 2025
        
        # Determine document type
        doc_type = "procurement_plan"
        if "monitoring" in filename.lower():
            doc_type = "monitoring_report"
        elif "supplemental" in filename.lower():
            doc_type = "supplemental_plan"
        
        metadata = {
            'doc_type': doc_type,
            'fiscal_year': fiscal_year,
            'department': 'DOH'
        }
        batch_metadata.append(metadata)
    
    # Process all documents
    results = await processor.process_multiple_documents(
        [str(pdf) for pdf in pdf_files],
        batch_metadata
    )
    
    # Report results
    successful = sum(1 for r in results if r.success)
    total_chunks = sum(r.chunk_count for r in results if r.success)
    total_tables = sum(r.table_count for r in results if r.success)
    
    logger.info(f"✅ Batch processing complete!")
    logger.info(f"📄 {successful}/{len(results)} documents processed successfully")
    logger.info(f"📊 Total chunks created: {total_chunks}")
    logger.info(f"🗂️ Total tables found: {total_tables}")
    
    return results


async def example_search():
    """Example: Search processed documents."""
    
    processor = PDFProcessor()
    
    # Example searches
    queries = [
        "annual procurement plan",
        "budget allocation",
        "medical supplies",
        "fiscal year 2024"
    ]
    
    for query in queries:
        logger.info(f"🔍 Searching for: '{query}'")
        
        results = await processor.search_documents(
            query=query,
            limit=5,
            filters={'doc_type': 'procurement_plan'}
        )
        
        if results:
            logger.info(f"Found {len(results)} results:")
            for i, result in enumerate(results[:3], 1):
                metadata = result['metadata']
                score = result['score']
                text_preview = metadata.get('text', '')[:100] + '...'
                logger.info(f"  {i}. Score: {score:.3f} - {text_preview}")
        else:
            logger.info("No results found")
        
        print()  # Empty line for readability


async def example_system_stats():
    """Example: Get system statistics."""
    
    processor = PDFProcessor()
    
    stats = await processor.get_system_stats()
    
    logger.info("📊 System Statistics:")
    logger.info(f"Vector Store: {stats.get('vector_store', {}).get('database_type', 'Unknown')}")
    logger.info(f"Total Vectors: {stats.get('vector_store', {}).get('total_vectors', 0)}")
    logger.info(f"OCR Engines Available: {stats.get('ocr_engines_available', [])}")
    logger.info(f"Embedding Model: {stats.get('configuration', {}).get('embedding_model', 'Unknown')}")


async def main():
    """Main example function."""
    
    logger.info("🚀 Starting Advanced PDF Chunking Pipeline Examples")
    
    try:
        # Set OpenAI API key (required for embeddings)
        if not os.getenv('OPENAI_API_KEY'):
            logger.warning("⚠️ OPENAI_API_KEY not set. Embedding generation will fail.")
            logger.info("Set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        
        # Example 1: Process single document
        logger.info("\n" + "="*50)
        logger.info("Example 1: Processing Single Document")
        logger.info("="*50)
        
        result = await example_single_document()
        
        if result and result.success:
            # Example 2: Search documents
            logger.info("\n" + "="*50)
            logger.info("Example 2: Searching Documents")
            logger.info("="*50)
            
            await example_search()
        
        # Example 3: Get system stats
        logger.info("\n" + "="*50)
        logger.info("Example 3: System Statistics")
        logger.info("="*50)
        
        await example_system_stats()
        
        # Example 4: Process multiple documents (commented out for demo)
        # logger.info("\n" + "="*50)
        # logger.info("Example 4: Processing Multiple Documents")
        # logger.info("="*50)
        # 
        # await example_multiple_documents()
        
    except Exception as e:
        logger.error(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("🏁 Examples completed!")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())