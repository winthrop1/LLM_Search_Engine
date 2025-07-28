"""
Test script focusing on the working search functionality.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Import our modules
from src.ingestion import DocumentIngestion
from src.preprocessing import TextPreprocessor
from src.indexing import IncrementalIndexer
from src.search import HybridSearcher

# Load environment variables
load_dotenv()

def test_search_functionality():
    """Test the core search functionality that's working."""
    print("🎯 Testing Enhanced Search Functionality")
    print("=" * 50)
    
    # Initialize core components
    print("🚀 Initializing components...")
    ingestion = DocumentIngestion("./data", "./processed")
    preprocessor = TextPreprocessor()
    indexer = IncrementalIndexer("./index", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Initialize hybrid searcher
    hybrid_searcher = HybridSearcher("sentence-transformers/all-MiniLM-L6-v2")
    
    print("✅ Components initialized")
    
    # Load existing index
    print("📚 Loading document index...")
    indexer.vector_store = indexer._load_vector_store()
    indexer.indexed_documents = indexer._load_indexed_documents()
    
    if indexer.indexed_documents:
        print(f"✅ Loaded {len(indexer.indexed_documents)} documents")
        
        # Index documents for hybrid search
        hybrid_searcher.index_documents(indexer.indexed_documents)
        print("✅ Hybrid search ready")
        
        # Test searches
        test_queries = [
            "financial performance",
            "revenue growth",
            "market conditions",
            "economic outlook",
            "investment strategy"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            
            # Basic semantic search
            semantic_results = indexer.search(query, k=2)
            print(f"📊 Semantic search: {len(semantic_results)} results")
            
            # Hybrid search
            hybrid_results = hybrid_searcher.search(query, top_k=2)
            print(f"🔀 Hybrid search: {len(hybrid_results)} results")
            
            if hybrid_results:
                result = hybrid_results[0]
                source = Path(result.document.metadata.get('source', 'Unknown')).name
                print(f"   Top result: {source}")
                print(f"   Combined Score: {result.combined_score:.3f}")
                print(f"   (Semantic: {result.semantic_score:.3f}, Keyword: {result.keyword_score:.3f})")
                if result.match_highlights:
                    print(f"   Keywords: {', '.join(result.match_highlights)}")
        
        # Show statistics
        print(f"\n📊 System Statistics:")
        index_stats = indexer.get_index_stats()
        for key, value in index_stats.items():
            print(f"   {key}: {value}")
        
        search_stats = hybrid_searcher.get_search_stats()
        print(f"\n🔍 Search Statistics:")
        for key, value in search_stats.items():
            print(f"   {key}: {value}")
            
    else:
        print("❌ No documents found in index")
    
    print(f"\n🎉 Search functionality test completed!")

if __name__ == "__main__":
    test_search_functionality()