"""
Incremental indexing module for smart document tracking and selective reprocessing.
Uses file hashing and metadata persistence for efficient updates.
"""

import os
import json
import pickle
import hashlib
from typing import List, Dict, Set, Optional, Tuple
from pathlib import Path
from datetime import datetime

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


class DocumentTracker:
    """Tracks document metadata for incremental indexing."""
    
    def __init__(self, metadata_file: str = "./processed/document_metadata.json"):
        """
        Initialize document tracker.
        
        Args:
            metadata_file: Path to metadata persistence file
        """
        self.metadata_file = Path(metadata_file)
        self.metadata_file.parent.mkdir(exist_ok=True)
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load existing metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading metadata: {e}")
        
        return {
            'documents': {},
            'last_updated': None,
            'index_version': 1
        }
    
    def _save_metadata(self):
        """Save metadata to file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving metadata: {e}")
    
    def get_file_info(self, file_path: Path) -> Dict:
        """Get current file information."""
        stat = file_path.stat()
        
        # Calculate file hash
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return {
            'path': str(file_path),
            'size': stat.st_size,
            'modified_time': stat.st_mtime,
            'hash': hash_sha256.hexdigest(),
            'last_processed': None
        }
    
    def is_document_changed(self, file_path: Path) -> bool:
        """Check if document has changed since last processing."""
        file_key = str(file_path)
        current_info = self.get_file_info(file_path)
        
        if file_key not in self.metadata['documents']:
            return True
        
        stored_info = self.metadata['documents'][file_key]
        
        # Check if hash or modification time changed
        return (current_info['hash'] != stored_info.get('hash') or
                current_info['modified_time'] != stored_info.get('modified_time'))
    
    def update_document_info(self, file_path: Path, processing_success: bool = True):
        """Update stored information for a document."""
        file_key = str(file_path)
        current_info = self.get_file_info(file_path)
        current_info['last_processed'] = datetime.now().isoformat()
        current_info['processing_success'] = processing_success
        
        self.metadata['documents'][file_key] = current_info
        self.metadata['last_updated'] = datetime.now().isoformat()
        self._save_metadata()
    
    def get_changed_documents(self, file_paths: List[Path]) -> Tuple[List[Path], List[Path]]:
        """
        Identify changed and new documents.
        
        Args:
            file_paths: List of file paths to check
            
        Returns:
            Tuple of (changed_files, deleted_files)
        """
        changed_files = []
        existing_files = set()
        
        for file_path in file_paths:
            existing_files.add(str(file_path))
            if self.is_document_changed(file_path):
                changed_files.append(file_path)
        
        # Find deleted files
        stored_files = set(self.metadata['documents'].keys())
        deleted_files = [Path(f) for f in stored_files - existing_files]
        
        return changed_files, deleted_files
    
    def remove_document_info(self, file_path: Path):
        """Remove information for deleted document."""
        file_key = str(file_path)
        if file_key in self.metadata['documents']:
            del self.metadata['documents'][file_key]
            self._save_metadata()
    
    def get_processing_stats(self) -> Dict:
        """Get processing statistics."""
        docs = self.metadata['documents']
        total_docs = len(docs)
        successful_docs = sum(1 for doc in docs.values() 
                            if doc.get('processing_success', False))
        
        return {
            'total_documents': total_docs,
            'successfully_processed': successful_docs,
            'last_updated': self.metadata.get('last_updated'),
            'index_version': self.metadata.get('index_version', 1)
        }


class IncrementalIndexer:
    """Handles incremental indexing with smart document tracking."""
    
    def __init__(self, 
                 index_dir: str = "./index",
                 embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize incremental indexer.
        
        Args:
            index_dir: Directory for storing index files
            embeddings_model: Model name for embeddings
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        
        self.tracker = DocumentTracker()
        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
        
        # Index file paths
        self.vector_store_path = self.index_dir / "vector_store"
        self.documents_path = self.index_dir / "documents.pkl"
        
        # Load existing index if available
        self.vector_store = self._load_vector_store()
        self.indexed_documents = self._load_indexed_documents()
    
    def _load_vector_store(self) -> Optional[FAISS]:
        """Load existing vector store if available."""
        if self.vector_store_path.exists():
            try:
                return FAISS.load_local(
                    str(self.vector_store_path), 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Error loading existing vector store: {e}")
        return None
    
    def _save_vector_store(self):
        """Save vector store to disk."""
        if self.vector_store:
            try:
                self.vector_store.save_local(str(self.vector_store_path))
            except Exception as e:
                print(f"Error saving vector store: {e}")
    
    def _load_indexed_documents(self) -> List[Document]:
        """Load list of indexed documents."""
        if self.documents_path.exists():
            try:
                with open(self.documents_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading indexed documents: {e}")
        return []
    
    def _save_indexed_documents(self):
        """Save list of indexed documents."""
        try:
            with open(self.documents_path, 'wb') as f:
                pickle.dump(self.indexed_documents, f)
        except Exception as e:
            print(f"Error saving indexed documents: {e}")
    
    def add_documents(self, documents: List[Document], file_path: Optional[Path] = None):
        """
        Add new documents to the index.
        
        Args:
            documents: List of documents to add
            file_path: Original file path for tracking
        """
        if not documents:
            return
        
        try:
            if self.vector_store is None:
                # Create new vector store
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
                self.indexed_documents = documents.copy()
            else:
                # Add to existing vector store
                self.vector_store.add_documents(documents)
                self.indexed_documents.extend(documents)
            
            # Update tracking information
            if file_path:
                self.tracker.update_document_info(file_path, True)
            
            print(f"Added {len(documents)} documents to index")
            
        except Exception as e:
            print(f"Error adding documents to index: {e}")
            if file_path:
                self.tracker.update_document_info(file_path, False)
    
    def remove_documents_by_source(self, source_path: str):
        """
        Remove documents from index by source path.
        
        Args:
            source_path: Source path to remove
        """
        # Filter out documents from the specified source
        remaining_docs = [doc for doc in self.indexed_documents 
                         if doc.metadata.get('source') != source_path]
        
        if len(remaining_docs) != len(self.indexed_documents):
            print(f"Removing {len(self.indexed_documents) - len(remaining_docs)} documents from {source_path}")
            
            # Rebuild vector store without the removed documents
            if remaining_docs:
                self.vector_store = FAISS.from_documents(remaining_docs, self.embeddings)
                self.indexed_documents = remaining_docs
            else:
                self.vector_store = None
                self.indexed_documents = []
            
            # Remove from tracking
            self.tracker.remove_document_info(Path(source_path))
    
    def update_index(self, new_documents: List[Document], 
                     changed_files: List[Path], 
                     deleted_files: List[Path]):
        """
        Update index with new, changed, and deleted documents.
        
        Args:
            new_documents: List of new/updated documents
            changed_files: List of files that changed
            deleted_files: List of files that were deleted
        """
        # Remove documents from deleted files
        for deleted_file in deleted_files:
            self.remove_documents_by_source(str(deleted_file))
        
        # Remove documents from changed files (they'll be re-added)
        for changed_file in changed_files:
            self.remove_documents_by_source(str(changed_file))
        
        # Add new/updated documents
        if new_documents:
            self.add_documents(new_documents)
        
        # Save everything
        self.save_index()
    
    def save_index(self):
        """Save index and metadata to disk."""
        self._save_vector_store()
        self._save_indexed_documents()
        
        print(f"Index saved with {len(self.indexed_documents)} documents")
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        Search the index.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        if self.vector_store is None:
            print("No index available for search")
            return []
        
        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def get_index_stats(self) -> Dict:
        """Get index statistics."""
        stats = self.tracker.get_processing_stats()
        stats.update({
            'indexed_documents': len(self.indexed_documents),
            'vector_store_available': self.vector_store is not None
        })
        return stats


def create_incremental_indexer(index_dir: str = "./index", 
                              embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> IncrementalIndexer:
    """
    Convenience function to create an incremental indexer.
    
    Args:
        index_dir: Directory for storing index files
        embeddings_model: Model name for embeddings
        
    Returns:
        IncrementalIndexer instance
    """
    return IncrementalIndexer(index_dir, embeddings_model)