"""
Test script to demonstrate the enhanced semantic search engine features.
"""

import os
from pathlib import Path
from src.ingestion import DocumentIngestion
from src.preprocessing import TextPreprocessor
from src.indexing import IncrementalIndexer

def test_multi_format_support():
    """Test multi-format document support."""
    print("=== Testing Multi-Format Document Support ===")
    
    ingestion = DocumentIngestion("./data")
    
    # Show supported formats
    formats = ingestion.get_supported_formats()
    print(f"Supported formats: {', '.join(formats)}")
    
    # Find documents
    documents = ingestion.find_documents()
    print(f"Found {len(documents)} documents in ./data/")
    
    for doc_path in documents:
        print(f"  - {doc_path.name} ({doc_path.suffix})")
    
    return documents

def test_text_preprocessing():
    """Test advanced text preprocessing."""
    print("\n=== Testing Text Preprocessing ===")
    
    # Sample text with various issues
    sample_text = """
    Page 1
    
    This is a docu-
    ment with hyphenated words    and    extra   spaces.
    
    
    There are multiple newlines above.
    
    Figure 1: This is a sample figure caption.
    
    | Name | Value | Description |
    |------|-------|-------------|
    | Item1| 100   | Sample data |
    
    Copyright © 2024 Company Name
    Printed on 2024-01-01
    """
    
    preprocessor = TextPreprocessor()
    
    # Test individual preprocessing steps
    print("Original text length:", len(sample_text))
    
    # Remove headers/footers
    no_headers = preprocessor.remove_headers_footers(sample_text)
    print("After header/footer removal:", len(no_headers))
    
    # Fix hyphenation
    fixed_hyphen = preprocessor.fix_hyphenation(no_headers)
    print("After hyphenation fix:", len(fixed_hyphen))
    
    # Normalize whitespace
    normalized = preprocessor.normalize_whitespace(fixed_hyphen)
    print("After whitespace normalization:", len(normalized))
    
    # Extract structured content
    structured = preprocessor.extract_tables_and_captions(normalized)
    print(f"Extracted {len(structured['tables'])} tables and {len(structured['captions'])} captions")
    
    print("\nProcessed text preview:")
    print(structured['cleaned_text'][:200] + "...")

def test_incremental_indexing():
    """Test incremental indexing functionality."""
    print("\n=== Testing Incremental Indexing ===")
    
    indexer = IncrementalIndexer("./index")
    
    # Get current stats
    stats = indexer.get_index_stats()
    print("Current index stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test document tracking
    tracker = indexer.tracker
    
    # Find test documents
    test_files = list(Path("./data").glob("*.pdf"))
    if test_files:
        test_file = test_files[0]
        print(f"\nTesting with file: {test_file}")
        
        # Check if document has changed
        has_changed = tracker.is_document_changed(test_file)
        print(f"Document changed since last processing: {has_changed}")
        
        # Get file info
        file_info = tracker.get_file_info(test_file)
        print(f"File hash: {file_info['hash'][:16]}...")
        print(f"File size: {file_info['size']} bytes")

def test_complete_pipeline():
    """Test the complete processing pipeline."""
    print("\n=== Testing Complete Pipeline ===")
    
    # Initialize components
    ingestion = DocumentIngestion("./data")
    preprocessor = TextPreprocessor()
    indexer = IncrementalIndexer("./index")
    
    # Find documents
    documents = ingestion.find_documents()
    if not documents:
        print("No documents found for testing")
        return
    
    # Process first document as example
    test_doc = documents[0]
    print(f"Processing: {test_doc}")
    
    try:
        # Load document
        loaded_docs = ingestion.load_document(test_doc)
        print(f"Loaded {len(loaded_docs)} document chunks")
        
        if loaded_docs:
            # Preprocess first chunk
            first_doc = loaded_docs[0]
            print(f"Original content length: {len(first_doc.page_content)}")
            
            processed_doc = preprocessor.preprocess_document(first_doc)
            print(f"Processed content length: {len(processed_doc.page_content)}")
            print(f"Processing steps: {processed_doc.metadata.get('preprocessing_steps', [])}")
            
            # Show content preview
            content_preview = processed_doc.page_content[:200].replace('\n', ' ')
            print(f"Content preview: {content_preview}...")
    
    except Exception as e:
        print(f"Error processing document: {e}")

def main():
    """Run all tests."""
    print("Enhanced Semantic Search Engine - Feature Tests")
    print("=" * 50)
    
    try:
        # Test individual components
        test_multi_format_support()
        test_text_preprocessing()
        test_incremental_indexing()
        test_complete_pipeline()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        print("\nTo use the full search engine, run:")
        print("  python main_enhanced.py")
        
    except Exception as e:
        print(f"Test error: {e}")
        print("\nNote: Some features may require additional dependencies.")
        print("Install with: pip install -r requirements.txt")

if __name__ == "__main__":
    main()