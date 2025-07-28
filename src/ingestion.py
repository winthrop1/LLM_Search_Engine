"""
Document ingestion module for multi-format document support.
Supports PDF, DOCX, TXT, MD, HTML, PPTX, XLSX, and image files.
Image formats (JPG, PNG, GIF, BMP, TIFF) are processed using OCR.
"""

import os
import glob
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .ocr import OCRProcessor, HAS_OCR


class ImageLoader:
    """Custom loader for image files using OCR."""
    
    def __init__(self, file_path: str, ocr_processor: Optional[OCRProcessor] = None):
        """
        Initialize ImageLoader.
        
        Args:
            file_path: Path to image file
            ocr_processor: OCR processor instance (optional)
        """
        self.file_path = file_path
        self.ocr_processor = ocr_processor or OCRProcessor()
        
        if not HAS_OCR or not HAS_PIL:
            raise ImportError(
                "Image processing requires OCR and PIL dependencies. "
                "Install with: pip install pytesseract pillow"
            )
    
    def load(self) -> List[Document]:
        """
        Load and process image using OCR.
        
        Returns:
            List containing single Document with OCR-extracted text
        """
        try:
            # Open image
            image = Image.open(self.file_path)
            
            # Extract text using OCR
            text = self.ocr_processor.extract_text_from_image(image)
            
            if not text.strip():
                print(f"No text found in image: {self.file_path}")
                return []
            
            # Create document
            document = Document(
                page_content=text,
                metadata={
                    'source': self.file_path,
                    'extraction_method': 'ocr',
                    'image_mode': image.mode,
                    'image_size': image.size
                }
            )
            
            return [document]
            
        except Exception as e:
            print(f"Error processing image {self.file_path}: {str(e)}")
            return []


class DocumentIngestion:
    """Handles multi-format document ingestion and processing."""
    
    SUPPORTED_EXTENSIONS = {
        '.pdf': PyPDFLoader,
        '.docx': UnstructuredWordDocumentLoader,
        '.txt': TextLoader,
        '.md': UnstructuredMarkdownLoader,
        '.html': UnstructuredHTMLLoader,
        '.pptx': UnstructuredPowerPointLoader,
        '.xlsx': UnstructuredExcelLoader,
        # Image formats (processed via OCR)
        '.jpg': ImageLoader,
        '.jpeg': ImageLoader,
        '.png': ImageLoader,
        '.gif': ImageLoader,
        '.bmp': ImageLoader,
        '.tiff': ImageLoader,
        '.tif': ImageLoader
    }
    
    def __init__(self, data_dir: str = "./data", processed_dir: str = "./processed"):
        """
        Initialize document ingestion.
        
        Args:
            data_dir: Directory containing raw documents
            processed_dir: Directory for processed documents
        """
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(exist_ok=True)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize OCR processor for image handling
        self.ocr_processor = None
        if HAS_OCR:
            try:
                self.ocr_processor = OCRProcessor()
            except Exception as e:
                print(f"OCR processor initialization failed: {e}")
                print("Image files will be skipped")
    
    def is_image_file(self, file_ext: str) -> bool:
        """Check if file extension is an image format."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif'}
        return file_ext.lower() in image_extensions
    
    def get_file_hash(self, file_path: Path) -> str:
        """Generate SHA-256 hash of file content for change detection."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def find_documents(self) -> List[Path]:
        """Find all supported documents in the data directory."""
        documents = []
        for ext in self.SUPPORTED_EXTENSIONS.keys():
            pattern = str(self.data_dir / f"**/*{ext}")
            documents.extend([Path(f) for f in glob.glob(pattern, recursive=True)])
        return documents
    
    def load_document(self, file_path: Path) -> List[Document]:
        """
        Load a single document using appropriate loader.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects
        """
        file_ext = file_path.suffix.lower()
        
        if file_ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {file_ext}")
        
        loader_class = self.SUPPORTED_EXTENSIONS[file_ext]
        
        try:
            # Handle different loader initialization patterns
            if self.is_image_file(file_ext):
                # Special handling for image files
                if not self.ocr_processor:
                    print(f"Skipping image file {file_path}: OCR not available")
                    return []
                loader = loader_class(str(file_path), self.ocr_processor)
            elif file_ext == '.txt':
                loader = loader_class(str(file_path), encoding='utf-8')
            else:
                loader = loader_class(str(file_path))
            
            documents = loader.load()
            
            # Add file metadata
            for doc in documents:
                doc.metadata.update({
                    'source': str(file_path),
                    'file_type': file_ext,
                    'file_hash': self.get_file_hash(file_path),
                    'file_size': file_path.stat().st_size,
                    'modified_time': file_path.stat().st_mtime
                })
            
            return documents
            
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return []
    
    def process_documents(self, file_paths: List[Path] = None) -> List[Document]:
        """
        Process multiple documents and split into chunks.
        
        Args:
            file_paths: List of file paths to process. If None, processes all files.
            
        Returns:
            List of processed Document chunks
        """
        if file_paths is None:
            file_paths = self.find_documents()
        
        all_documents = []
        
        for file_path in file_paths:
            print(f"Processing: {file_path}")
            
            try:
                # Load document
                documents = self.load_document(file_path)
                
                if not documents:
                    continue
                
                # Split documents into chunks
                for doc in documents:
                    chunks = self.text_splitter.split_documents([doc])
                    all_documents.extend(chunks)
                    
            except Exception as e:
                print(f"Failed to process {file_path}: {str(e)}")
                continue
        
        print(f"Processed {len(file_paths)} files into {len(all_documents)} chunks")
        return all_documents
    
    def get_supported_formats(self) -> List[str]:
        """Return list of supported file formats."""
        return list(self.SUPPORTED_EXTENSIONS.keys())


def load_documents_from_folder(folder_path: str) -> List[Document]:
    """
    Convenience function to load all documents from a folder.
    
    Args:
        folder_path: Path to folder containing documents
        
    Returns:
        List of processed Document chunks
    """
    ingestion = DocumentIngestion(data_dir=folder_path)
    return ingestion.process_documents()