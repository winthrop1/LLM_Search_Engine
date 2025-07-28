"""
Document ingestion module for multi-format document support.
Supports PDF, DOCX, TXT, MD, HTML, PPTX, and XLSX files.
"""

import os
import glob
import hashlib
from typing import List, Dict, Any
from pathlib import Path

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


class DocumentIngestion:
    """Handles multi-format document ingestion and processing."""
    
    SUPPORTED_EXTENSIONS = {
        '.pdf': PyPDFLoader,
        '.docx': UnstructuredWordDocumentLoader,
        '.txt': TextLoader,
        '.md': UnstructuredMarkdownLoader,
        '.html': UnstructuredHTMLLoader,
        '.pptx': UnstructuredPowerPointLoader,
        '.xlsx': UnstructuredExcelLoader
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
            if file_ext == '.txt':
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