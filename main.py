"""
Enhanced semantic document search engine with multi-format support, OCR, and incremental indexing.
"""

import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
# Import our custom modules
from src.ingestion import DocumentIngestion
from src.preprocessing import TextPreprocessor
from src.ocr import extract_text_with_ocr_fallback
from src.indexing import IncrementalIndexer
from src.search import HybridSearcher
from src.llm_router import LLMRouter
from src.conversation import ConversationManager

# Load environment variables
load_dotenv()

class SemanticSearchEngine:
    """Enhanced semantic search engine with multi-format support."""
    
    def __init__(self, 
                 data_dir: str = "./data",
                 processed_dir: str = "./processed", 
                 index_dir: str = "./index",
                 embeddings_model: str = None):
        """
        Initialize the semantic search engine.
        
        Args:
            data_dir: Directory containing raw documents
            processed_dir: Directory for processed documents  
            index_dir: Directory for index storage
            embeddings_model: Model name for embeddings
        """
        # Get embeddings model from environment or use default
        self.embeddings_model = (embeddings_model or 
                                os.getenv("VECTOR_DB_NAME") or 
                                "sentence-transformers/all-MiniLM-L6-v2")
        
        # Initialize components
        self.ingestion = DocumentIngestion(data_dir, processed_dir)
        self.preprocessor = TextPreprocessor()
        self.indexer = IncrementalIndexer(index_dir, self.embeddings_model)
        
        # Initialize advanced features
        self.hybrid_searcher = HybridSearcher(self.embeddings_model)
        self.llm_router = None
        self.conversation_manager = None
        
        print(f"Initialized with embeddings model: {self.embeddings_model}")
        print(f"Supported formats: {', '.join(self.ingestion.get_supported_formats())}")
        
        # Try to setup LLM features
        self._try_setup_llm()
    
    def _try_setup_llm(self):
        """Try to setup LLM features."""
        try:
            self.llm_router = LLMRouter()
            self.conversation_manager = ConversationManager(
                self.llm_router, 
                self.hybrid_searcher
            )
            
            provider_info = self.llm_router.get_provider_info()
            print(f"🤖 AI Features Available:")
            print(f"   Provider: {provider_info['provider']}")
            print(f"   Model: {provider_info['model']}")
            print(f"   Streaming: {'🌊 Enabled' if provider_info['streaming_enabled'] else '🚫 Disabled'}")
            
        except Exception as e:
            print(f"⚠️  AI features unavailable: {e}")
            print("💡 You can still use enhanced search features")
    
    def build_index(self, force_rebuild: bool = False):
        """
        Build or update the document index.
        
        Args:
            force_rebuild: If True, rebuilds entire index from scratch
        """
        print("Building/updating document index...")
        
        # Find all documents
        all_files = self.ingestion.find_documents()
        print(f"Found {len(all_files)} documents")
        
        if not all_files:
            print("No documents found to index")
            return
        
        if force_rebuild:
            print("Force rebuilding entire index...")
            # Process all documents
            documents = self.ingestion.process_documents(all_files)
            
            # Preprocess documents
            documents = self.preprocessor.preprocess_documents(documents)
            
            # Clear existing index and rebuild
            self.indexer.indexed_documents = []
            self.indexer.vector_store = None
            self.indexer.add_documents(documents)
            
            # Update tracking for all files
            for file_path in all_files:
                self.indexer.tracker.update_document_info(file_path, True)
        else:
            # Incremental update
            changed_files, deleted_files = self.indexer.tracker.get_changed_documents(all_files)
            
            print(f"Changed files: {len(changed_files)}")
            print(f"Deleted files: {len(deleted_files)}")
            
            if not changed_files and not deleted_files:
                print("No changes detected - index is up to date")
                # Still need to setup hybrid searcher with existing documents
                if self.indexer.indexed_documents:
                    self.hybrid_searcher.index_documents(self.indexer.indexed_documents)
                return
            
            # Process only changed documents
            new_documents = []
            for file_path in changed_files:
                print(f"Processing changed file: {file_path}")
                
                try:
                    # Load and preprocess document
                    docs = self.ingestion.process_documents([file_path])
                    
                    # Apply OCR fallback for PDFs if needed
                    if file_path.suffix.lower() == '.pdf':
                        docs = extract_text_with_ocr_fallback(file_path, docs)
                    
                    # Preprocess
                    docs = self.preprocessor.preprocess_documents(docs)
                    new_documents.extend(docs)
                    
                    # Update tracking
                    self.indexer.tracker.update_document_info(file_path, True)
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    self.indexer.tracker.update_document_info(file_path, False)
            
            # Update index
            self.indexer.update_index(new_documents, changed_files, deleted_files)
        
        # Save index
        self.indexer.save_index()
        
        # Setup hybrid searcher if we have documents
        if self.indexer.indexed_documents:
            self.hybrid_searcher.index_documents(self.indexer.indexed_documents)
        
        # Print statistics
        stats = self.indexer.get_index_stats()
        print(f"\nIndex Statistics:")
        print(f"- Total documents in index: {stats['indexed_documents']}")
        print(f"- Total files tracked: {stats['total_documents']}")
        print(f"- Successfully processed: {stats['successfully_processed']}")
        print(f"- Last updated: {stats['last_updated']}")
    
    def search(self, query: str, k: int = 5, show_metadata: bool = True) -> list:
        """
        Perform semantic search.
        
        Args:
            query: Search query
            k: Number of results to return
            show_metadata: Whether to show document metadata
            
        Returns:
            List of search results
        """
        if self.indexer.vector_store is None:
            print("No index available. Please build the index first.")
            return []
        
        print(f"Searching for: '{query}'\n")
        
        results = self.indexer.search(query, k)
        
        for i, doc in enumerate(results, 1):
            print(f"Result {i}:")
            print("-" * 50)
            
            # Show source information
            source = doc.metadata.get('source', 'Unknown')
            file_type = doc.metadata.get('file_type', 'unknown')
            print(f"Source: {Path(source).name} ({file_type})")
            
            if show_metadata:
                # Show additional metadata
                if 'page' in doc.metadata:
                    print(f"Page: {doc.metadata['page']}")
                if 'extraction_method' in doc.metadata:
                    print(f"Extraction: {doc.metadata['extraction_method']}")
                if 'processing_steps' in doc.metadata:
                    print(f"Processing: {', '.join(doc.metadata['processing_steps'])}")
            
            print("\nContent:")
            content = doc.page_content.strip()
            if len(content) > 500:
                content = content[:500] + "..."
            print(content)
            print("\n")
        
        return results
    
    def get_index_info(self):
        """Display information about the current index."""
        stats = self.indexer.get_index_stats()
        
        print("=== Index Information ===")
        print(f"Indexed documents: {stats['indexed_documents']}")
        print(f"Total tracked files: {stats['total_documents']}")
        print(f"Successfully processed: {stats['successfully_processed']}")
        print(f"Vector store available: {stats['vector_store_available']}")
        print(f"Last updated: {stats['last_updated']}")
        print(f"Index version: {stats['index_version']}")
        
        # Show supported formats
        print(f"\nSupported formats: {', '.join(self.ingestion.get_supported_formats())}")
    
    def _show_commands(self):
        """Display available commands and current status."""
        print("\nCommands:")
        print("  /info     - Show index and system information") 
        print("  /rebuild  - Force rebuild entire index")
        print("  /update   - Update index (incremental)")
        print("  /hybrid   - Use hybrid search (semantic + keyword)")
        
        if self.conversation_manager:
            print("  /ai       - Ask AI question with conversation memory")
            print("  /stream   - Toggle streaming responses")
            print("  /conv     - Show conversation info")
        
        print("  /quit     - Exit search")
        print("  <query>   - Basic semantic search")
        print()
        
        # Show current streaming status if AI available
        if self.llm_router:
            streaming_status = "🌊 Enabled" if self.llm_router.use_streaming else "🚫 Disabled"
            print(f"Streaming: {streaming_status}")
            print()

    def interactive_search(self):
        """Start interactive search session."""
        print("=== Enhanced Semantic Document Search Engine ===")
        print("Multi-format support, OCR, hybrid search, and AI conversations")
        self._show_commands()
        
        while True:
            try:
                user_input = input("Search> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['/quit', '/exit']:
                    print("Goodbye!")
                    break
                elif user_input.lower() == '/info':
                    self.get_index_info()
                    self._show_commands()
                elif user_input.lower() == '/rebuild':
                    self.build_index(force_rebuild=True)
                    self._show_commands()
                elif user_input.lower() == '/update':
                    self.build_index(force_rebuild=False)
                    self._show_commands()
                elif user_input.lower() == '/hybrid':
                    query = input("Hybrid search query> ").strip()
                    if query:
                        self._hybrid_search(query)
                    self._show_commands()
                elif user_input.lower() == '/ai' and self.conversation_manager:
                    query = input("AI question> ").strip()
                    if query:
                        asyncio.run(self._ai_question(query))
                    self._show_commands()
                elif user_input.lower() == '/stream' and self.llm_router:
                    self._toggle_streaming()
                    self._show_commands()
                elif user_input.lower() == '/conv' and self.conversation_manager:
                    self._show_conversation_info()
                    self._show_commands()
                else:
                    # Perform basic semantic search
                    self.search(user_input)
                    self._show_commands()
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _hybrid_search(self, query: str):
        """Perform hybrid search."""
        print(f"🔍 Hybrid search for: '{query}'")
        results = self.hybrid_searcher.search(query, top_k=5)
        
        if not results:
            print("No results found.")
            return
        
        print(f"\n📋 Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            source = Path(result.document.metadata.get('source', 'Unknown')).name
            content = result.document.page_content[:200] + "..." if len(result.document.page_content) > 200 else result.document.page_content
            
            print(f"\n{i}. {source}")
            print(f"   Combined Score: {result.combined_score:.3f}")
            print(f"   (Semantic: {result.semantic_score:.3f}, Keyword: {result.keyword_score:.3f})")
            if result.match_highlights:
                print(f"   Keywords: {', '.join(result.match_highlights)}")
            print(f"   Content: {content}")
    
    async def _ai_question(self, query: str):
        """Ask AI question with conversation memory."""
        print(f"🤖 AI Question: '{query}'")
        try:
            response, search_results = await self.conversation_manager.ask_question(query)
            print(f"\n📝 AI Response:")
            print(response)
            print(f"\n📚 Based on {len(search_results)} source documents")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def _toggle_streaming(self):
        """Toggle streaming responses."""
        self.llm_router.use_streaming = not self.llm_router.use_streaming
        status = "🌊 Enabled" if self.llm_router.use_streaming else "🚫 Disabled"
        print(f"Streaming responses: {status}")
    
    def _show_conversation_info(self):
        """Show conversation information."""
        if not self.conversation_manager.current_conversation:
            print("No active conversation")
            return
        
        summary = self.conversation_manager.get_conversation_summary()
        print(f"📊 Conversation Info:")
        print(f"   ID: {summary['conversation_id']}")
        print(f"   Turns: {summary['total_turns']}")
        print(f"   Tokens: {summary['total_tokens']}")
        print(f"   Provider: {summary['provider']}")
        print(f"   Created: {summary['created_at']}")


def main():
    """Main entry point."""
    # Configuration from environment or defaults
    data_dir = os.getenv("DATA_FOLDER", "./data")
    processed_dir = "./processed"
    index_dir = "./index"
    
    # Initialize search engine
    engine = SemanticSearchEngine(data_dir, processed_dir, index_dir)
    
    # Build initial index if needed
    stats = engine.indexer.get_index_stats()
    if stats['indexed_documents'] == 0:
        print("No existing index found. Building initial index...")
        engine.build_index()
    else:
        print(f"Existing index found with {stats['indexed_documents']} documents")
        print("Checking for updates...")
        engine.build_index()  # Incremental update
    
    # Start interactive search
    engine.interactive_search()


if __name__ == "__main__":
    main()