"""
Test Script for Advanced PDF Chunking Pipeline

This script performs basic validation of the system components
without requiring external dependencies like OpenAI API or vector databases.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all modules can be imported successfully."""
    try:
        logger.info("Testing imports...")
        
        # Test basic imports
        from src.models import (
            DocumentMetadata, ConfigParameters, ContentType, 
            OCREngine, PDFType, ProcessingResult
        )
        logger.info("✅ Models imported successfully")
        
        from src.metadata_extractor import MetadataExtractor
        logger.info("✅ MetadataExtractor imported successfully")
        
        from src.ocr_merger import OCRMerger
        logger.info("✅ OCRMerger imported successfully")
        
        from src.section_tagger import SectionTagger
        logger.info("✅ SectionTagger imported successfully")
        
        from src.section_aware_chunker import SectionAwareChunker
        logger.info("✅ SectionAwareChunker imported successfully")
        
        # These may fail if dependencies are not installed
        try:
            from src.ocr_engines import OCREngineManager
            logger.info("✅ OCREngineManager imported successfully")
        except ImportError as e:
            logger.warning(f"⚠️ OCREngineManager import failed: {e}")
        
        try:
            from src.sentence_chunker import SentenceChunker
            logger.info("✅ SentenceChunker imported successfully")
        except ImportError as e:
            logger.warning(f"⚠️ SentenceChunker import failed: {e}")
        
        try:
            from src.embedding_generator import EmbeddingGenerator
            logger.info("✅ EmbeddingGenerator imported successfully")
        except ImportError as e:
            logger.warning(f"⚠️ EmbeddingGenerator import failed: {e}")
        
        try:
            from src.vector_store import VectorStoreManager
            logger.info("✅ VectorStoreManager imported successfully")
        except ImportError as e:
            logger.warning(f"⚠️ VectorStoreManager import failed: {e}")
        
        try:
            from src.pdf_processor import PDFProcessor
            logger.info("✅ PDFProcessor imported successfully")
        except ImportError as e:
            logger.warning(f"⚠️ PDFProcessor import failed: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Import test failed: {e}")
        return False


def test_model_creation():
    """Test creating model instances."""
    try:
        logger.info("Testing model creation...")
        
        from src.models import (
            DocumentMetadata, ConfigParameters, ContentType,
            OCREngine, PDFType
        )
        
        # Test ConfigParameters
        config = ConfigParameters()
        logger.info(f"✅ ConfigParameters created with {len(config.ocr_engines)} OCR engines")
        
        # Test DocumentMetadata
        metadata = DocumentMetadata(
            pdf_type=PDFType.DIGITAL,
            page_count=10,
            file_size=1024000,
            file_path="/test/document.pdf"
        )
        logger.info(f"✅ DocumentMetadata created for {metadata.page_count} page document")
        
        # Test enums
        assert ContentType.HEADER_H1 in ContentType
        assert OCREngine.TESSERACT in OCREngine
        assert PDFType.DIGITAL in PDFType
        logger.info("✅ Enums working correctly")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model creation test failed: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality of key components."""
    try:
        logger.info("Testing basic functionality...")
        
        from src.models import ConfigParameters
        from src.metadata_extractor import MetadataExtractor
        from src.ocr_merger import OCRMerger
        from src.section_tagger import SectionTagger
        
        # Test MetadataExtractor initialization
        extractor = MetadataExtractor()
        logger.info("✅ MetadataExtractor initialized")
        
        # Test OCRMerger initialization
        config = ConfigParameters()
        merger = OCRMerger(config.model_dump())
        logger.info("✅ OCRMerger initialized")
        
        # Test SectionTagger initialization
        tagger = SectionTagger(config.model_dump())
        logger.info("✅ SectionTagger initialized")
        
        # Test SectionAwareChunker initialization
        from src.section_aware_chunker import SectionAwareChunker
        chunker = SectionAwareChunker(config)
        logger.info("✅ SectionAwareChunker initialized")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic functionality test failed: {e}")
        return False


def test_configuration():
    """Test configuration loading and validation."""
    try:
        logger.info("Testing configuration...")
        
        from src.models import ConfigParameters, OCREngine
        
        # Test default configuration
        config = ConfigParameters()
        assert config.target_chunk_size > 0
        assert config.embedding_model is not None
        assert len(config.ocr_engines) > 0
        logger.info("✅ Default configuration valid")
        
        # Test custom configuration
        custom_config = ConfigParameters(
            target_chunk_size=800,
            max_chunk_size=1000,
            ocr_engines=[OCREngine.PYPDF2],
            embedding_model="text-embedding-ada-002"
        )
        assert custom_config.target_chunk_size == 800
        assert custom_config.ocr_engines == [OCREngine.PYPDF2]
        logger.info("✅ Custom configuration working")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False


def test_file_structure():
    """Test that all required files are present."""
    try:
        logger.info("Testing file structure...")
        
        src_dir = Path(__file__).parent / 'src'
        required_files = [
            '__init__.py',
            'models.py',
            'metadata_extractor.py',
            'ocr_engines.py',
            'ocr_merger.py',
            'sentence_chunker.py',
            'section_tagger.py',
            'section_aware_chunker.py',
            'embedding_generator.py',
            'vector_store.py',
            'pdf_processor.py'
        ]
        
        missing_files = []
        for file in required_files:
            if not (src_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            logger.error(f"❌ Missing files: {missing_files}")
            return False
        
        logger.info("✅ All required files present")
        
        # Check config file
        config_file = Path(__file__).parent / 'config.yaml'
        if config_file.exists():
            logger.info("✅ Configuration file present")
        else:
            logger.warning("⚠️ Configuration file missing")
        
        # Check requirements file
        req_file = Path(__file__).parent / 'requirements.txt'
        if req_file.exists():
            logger.info("✅ Requirements file present")
        else:
            logger.warning("⚠️ Requirements file missing")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ File structure test failed: {e}")
        return False


def run_all_tests():
    """Run all validation tests."""
    logger.info("🚀 Starting Advanced PDF Pipeline Validation Tests")
    logger.info("="*60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Module Imports", test_imports),
        ("Model Creation", test_model_creation),
        ("Configuration", test_configuration),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} Test...")
        logger.info("-" * 40)
        
        try:
            result = test_func()
            results[test_name] = result
            
            if result:
                logger.info(f"✅ {test_name} test PASSED")
            else:
                logger.error(f"❌ {test_name} test FAILED")
                
        except Exception as e:
            logger.error(f"💥 {test_name} test CRASHED: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📊 TEST SUMMARY")
    logger.info("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:20} : {status}")
    
    logger.info("-" * 60)
    logger.info(f"Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is ready for use.")
    elif passed >= total * 0.8:
        logger.info("⚠️ Most tests passed. Some optional dependencies may be missing.")
    else:
        logger.warning("🚨 Multiple test failures. Check dependencies and installation.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)