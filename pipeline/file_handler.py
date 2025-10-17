"""
File System Handler - Stage 1 of PDF Chunking Pipeline

Handles OS detection, cross-platform paths, temp directories, and file operations.
"""

import platform
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List
import logging
import os
import stat

logger = logging.getLogger(__name__)

class FileSystemHandler:
    """Cross-platform file system handler with OS detection and pathlib integration."""
    
    def __init__(self):
        self.os_type = platform.system()  # Windows, Linux, Darwin
        self.temp_dir = None
        self.output_dir = None
        self.temp_files = []
        
        logger.info(f"Detected OS: {self.os_type}")
    
    def setup(self, output_dir: Optional[str] = None):
        """Initialize file system handler with directories."""
        try:
            # Create OS-appropriate temp directory
            self.temp_dir = Path(tempfile.mkdtemp(prefix="pdf_pipeline_"))
            logger.info(f"Created temp directory: {self.temp_dir}")
            
            # Set up output directory
            if output_dir:
                self.output_dir = Path(output_dir)
            else:
                self.output_dir = Path.cwd() / "output"
            
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory: {self.output_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup file system: {e}")
            return False
    
    def detect_os(self) -> str:
        """Detect operating system."""
        return self.os_type
    
    def create_temp_dir(self, prefix: str = "pdf_temp_") -> Path:
        """Create a temporary directory with given prefix."""
        temp_path = Path(tempfile.mkdtemp(prefix=prefix))
        self.temp_files.append(temp_path)
        return temp_path
    
    def validate_pdf(self, pdf_path: Path) -> bool:
        """Validate if file is a readable PDF."""
        try:
            pdf_path = Path(pdf_path)
            
            # Check if file exists
            if not pdf_path.exists():
                logger.error(f"PDF file does not exist: {pdf_path}")
                return False
            
            # Check if it's a file (not directory)
            if not pdf_path.is_file():
                logger.error(f"Path is not a file: {pdf_path}")
                return False
            
            # Check file extension
            if pdf_path.suffix.lower() != '.pdf':
                logger.warning(f"File doesn't have .pdf extension: {pdf_path}")
            
            # Check file size
            file_size = pdf_path.stat().st_size
            if file_size == 0:
                logger.error(f"PDF file is empty: {pdf_path}")
                return False
            
            # Check if file is readable
            if not os.access(pdf_path, os.R_OK):
                logger.error(f"PDF file is not readable: {pdf_path}")
                return False
            
            logger.info(f"PDF validation successful: {pdf_path} ({file_size} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"Error validating PDF {pdf_path}: {e}")
            return False
    
    def get_output_path(self, filename: str, subfolder: str = "") -> Path:
        """Get output path for processed files."""
        if subfolder:
            output_path = self.output_dir / subfolder
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = self.output_dir
        
        return output_path / filename
    
    def handle_file_permissions(self, path: Path) -> bool:
        """Handle file permissions across different OS."""
        try:
            path = Path(path)
            
            if self.os_type == "Windows":
                # Windows-specific permission handling
                if not os.access(path, os.R_OK):
                    logger.warning(f"File not readable on Windows: {path}")
                    return False
            
            elif self.os_type in ["Linux", "Darwin"]:
                # Unix-like permission handling
                current_permissions = path.stat().st_mode
                
                # Ensure read permissions
                if not (current_permissions & stat.S_IRUSR):
                    try:
                        path.chmod(current_permissions | stat.S_IRUSR)
                        logger.info(f"Added read permissions to: {path}")
                    except PermissionError:
                        logger.error(f"Cannot modify permissions for: {path}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling permissions for {path}: {e}")
            return False
    
    def cleanup_temp_files(self):
        """Clean up all temporary files and directories."""
        cleaned_count = 0
        
        try:
            # Clean up main temp directory
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temp directory: {self.temp_dir}")
                cleaned_count += 1
            
            # Clean up additional temp files
            for temp_path in self.temp_files:
                if temp_path.exists():
                    if temp_path.is_dir():
                        shutil.rmtree(temp_path)
                    else:
                        temp_path.unlink()
                    cleaned_count += 1
            
            self.temp_files.clear()
            logger.info(f"Cleaned up {cleaned_count} temporary items")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get_safe_filename(self, filename: str) -> str:
        """Generate OS-safe filename."""
        # Remove or replace problematic characters
        if self.os_type == "Windows":
            # Windows has more restrictive filename rules
            invalid_chars = '<>:"/\\|?*'
            for char in invalid_chars:
                filename = filename.replace(char, '_')
        else:
            # Unix-like systems
            filename = filename.replace('/', '_')
        
        # Ensure filename isn't too long
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255-len(ext)] + ext
        
        return filename
    
    def copy_file_safe(self, src: Path, dst: Path) -> bool:
        """Safely copy file with error handling."""
        try:
            src_path = Path(src)
            dst_path = Path(dst)
            
            # Ensure destination directory exists
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(src_path, dst_path)
            
            # Verify copy
            if dst_path.exists() and dst_path.stat().st_size == src_path.stat().st_size:
                logger.info(f"Successfully copied: {src_path} -> {dst_path}")
                return True
            else:
                logger.error(f"Copy verification failed: {src_path} -> {dst_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error copying file {src} -> {dst}: {e}")
            return False
    
    def get_file_info(self, file_path: Path) -> dict:
        """Get comprehensive file information."""
        try:
            path = Path(file_path)
            stat_info = path.stat()
            
            return {
                'path': str(path.absolute()),
                'name': path.name,
                'stem': path.stem,
                'suffix': path.suffix,
                'size_bytes': stat_info.st_size,
                'size_mb': round(stat_info.st_size / (1024 * 1024), 2),
                'created': stat_info.st_ctime,
                'modified': stat_info.st_mtime,
                'is_readable': os.access(path, os.R_OK),
                'is_writable': os.access(path, os.W_OK),
                'permissions': oct(stat_info.st_mode)[-3:],
                'parent_dir': str(path.parent)
            }
            
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {e}")
            return {}
    
    def __del__(self):
        """Cleanup on object destruction."""
        try:
            self.cleanup_temp_files()
        except:
            pass  # Ignore errors during cleanup