"""
Enhanced Semantic Document Search Engine Demo
Showcases all features: hybrid search, LLM integration, conversation, and summarization.
"""

import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Import our modules
from src.ingestion import DocumentIngestion
from src.preprocessing import TextPreprocessor
from src.indexing import IncrementalIndexer
from src.search import HybridSearcher
from src.llm_router import LLMRouter
from src.conversation import ConversationManager
from src.summarization import SummaryManager, SummaryStyle, SummaryLength

# Load environment variables
load_dotenv()


class EnhancedSemanticEngine:
    """Enhanced semantic search engine with all advanced features."""
    
    def __init__(self, 
                 data_dir: str = "./data",
                 processed_dir: str = "./processed",
                 index_dir: str = "./index",
                 embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the enhanced engine."""
        
        print("🚀 Initializing Enhanced Semantic Document Search Engine...")
        
        # Core components
        self.ingestion = DocumentIngestion(data_dir, processed_dir)
        self.preprocessor = TextPreprocessor()
        self.indexer = IncrementalIndexer(index_dir, embeddings_model)
        
        # Advanced search
        self.hybrid_searcher = HybridSearcher(embeddings_model)
        
        # LLM integration
        self.llm_router = None
        self.conversation_manager = None
        self.summary_manager = None
        
        print(f"✅ Core components initialized")
        print(f"📁 Data directory: {data_dir}")
        print(f"🔧 Supported formats: {', '.join(self.ingestion.get_supported_formats())}")
    
    def setup_llm(self, provider: str = None, model: str = None, api_key: str = None):
        """Setup LLM components."""
        try:
            print(f"🤖 Setting up LLM integration...")
            
            # Initialize LLM router
            self.llm_router = LLMRouter(provider, model, api_key)
            
            # Initialize conversation manager
            self.conversation_manager = ConversationManager(
                self.llm_router, 
                self.hybrid_searcher
            )
            
            # Initialize summary manager
            self.summary_manager = SummaryManager(self.llm_router)
            
            provider_info = self.llm_router.get_provider_info()
            print(f"✅ LLM setup complete:")
            print(f"   Provider: {provider_info['provider']}")
            print(f"   Model: {provider_info['model']}")
            print(f"   Max Tokens: {provider_info['max_tokens']}")
            print(f"   Streaming: {'🌊 Enabled' if provider_info['streaming_enabled'] else '🚫 Disabled'}")
            
            return True
            
        except Exception as e:
            print(f"❌ LLM setup failed: {e}")
            print("💡 You can still use basic search without LLM features")
            return False
    
    def build_index(self, force_rebuild: bool = False):
        """Build or update the document index."""
        print("\n📚 Building/updating document index...")
        
        # Find all documents
        all_files = self.ingestion.find_documents()
        print(f"📄 Found {len(all_files)} documents")
        
        if not all_files:
            print("⚠️  No documents found to index")
            return False
        
        # Process documents (existing logic from main_enhanced.py)
        if force_rebuild:
            print("🔄 Force rebuilding entire index...")
            documents = self.ingestion.process_documents(all_files)
            documents = self.preprocessor.preprocess_documents(documents)
            
            self.indexer.indexed_documents = []
            self.indexer.vector_store = None
            self.indexer.add_documents(documents)
            
            for file_path in all_files:
                self.indexer.tracker.update_document_info(file_path, True)
        else:
            # Incremental update
            changed_files, deleted_files = self.indexer.tracker.get_changed_documents(all_files)
            
            if not changed_files and not deleted_files:
                print("✅ Index is up to date")
            else:
                print(f"🔄 Processing {len(changed_files)} changed files")
                new_documents = []
                
                for file_path in changed_files:
                    try:
                        docs = self.ingestion.process_documents([file_path])
                        docs = self.preprocessor.preprocess_documents(docs)
                        new_documents.extend(docs)
                        self.indexer.tracker.update_document_info(file_path, True)
                    except Exception as e:
                        print(f"❌ Error processing {file_path}: {e}")
                
                self.indexer.update_index(new_documents, changed_files, deleted_files)
        
        # Save index and setup hybrid search
        self.indexer.save_index()
        
        # Index documents for hybrid search
        if self.indexer.indexed_documents:
            self.hybrid_searcher.index_documents(self.indexer.indexed_documents)
        
        # Print statistics
        stats = self.indexer.get_index_stats()
        print(f"✅ Index built successfully:")
        print(f"   📊 Total documents: {stats['indexed_documents']}")
        print(f"   📈 Successfully processed: {stats['successfully_processed']}")
        print(f"   🕒 Last updated: {stats['last_updated']}")
        
        return True
    
    def demo_basic_search(self):
        """Demonstrate basic semantic search."""
        print("\n" + "="*60)
        print("🔍 DEMO: Basic Semantic Search")
        print("="*60)
        
        query = "financial performance and revenue"
        print(f"Query: '{query}'")
        
        # Traditional semantic search
        results = self.indexer.search(query, k=3)
        
        print(f"\n📋 Found {len(results)} results:")
        for i, doc in enumerate(results, 1):
            source = Path(doc.metadata.get('source', 'Unknown')).name
            content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            print(f"\n{i}. {source}")
            print(f"   {content}")
    
    def demo_hybrid_search(self):
        """Demonstrate hybrid search combining semantic and keyword approaches."""
        print("\n" + "="*60)
        print("🔍 DEMO: Hybrid Search (Semantic + Keyword)")
        print("="*60)
        
        query = "revenue growth financial performance"
        print(f"Query: '{query}'")
        
        # Hybrid search
        results = self.hybrid_searcher.search(query, top_k=3)
        
        print(f"\n📋 Found {len(results)} hybrid search results:")
        for i, result in enumerate(results, 1):
            source = Path(result.document.metadata.get('source', 'Unknown')).name
            content = result.document.page_content[:200] + "..." if len(result.document.page_content) > 200 else result.document.page_content
            
            print(f"\n{i}. {source}")
            print(f"   Semantic Score: {result.semantic_score:.3f}")
            print(f"   Keyword Score: {result.keyword_score:.3f}")
            print(f"   Combined Score: {result.combined_score:.3f}")
            print(f"   Keywords: {', '.join(result.match_highlights)}")
            print(f"   Content: {content}")
    
    async def demo_conversation(self):
        """Demonstrate conversational AI with memory."""
        if not self.conversation_manager:
            print("\n⚠️  LLM not available - skipping conversation demo")
            return
        
        print("\n" + "="*60)
        print("💬 DEMO: Conversational AI with Memory")
        print("="*60)
        
        # Check streaming status
        streaming_status = "🌊 Streaming Enabled" if self.llm_router.use_streaming else "🚫 Streaming Disabled"
        print(f"Response Mode: {streaming_status}")
        
        # Start a new conversation
        conv_id = self.conversation_manager.start_new_conversation()
        print(f"Started conversation: {conv_id}")
        
        # Ask a series of related questions
        questions = [
            "What is the main financial performance discussed in the documents?",
            "Can you provide more details about the revenue figures?",
            "How does this compare to previous periods?"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n💭 Question {i}: {question}")
            print("🤖 Generating response...")
            
            try:
                response, search_results = await self.conversation_manager.ask_question(
                    question, search_top_k=3
                )
                
                print(f"📝 Response: {response}")
                print(f"📚 Used {len(search_results)} source documents")
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Show conversation summary
        summary = self.conversation_manager.get_conversation_summary()
        print(f"\n📊 Conversation Summary:")
        print(f"   Turns: {summary['total_turns']}")
        print(f"   Tokens: {summary['total_tokens']}")
        print(f"   Provider: {summary['provider']}")
    
    async def demo_summarization(self):
        """Demonstrate document summarization."""
        if not self.summary_manager:
            print("\n⚠️  LLM not available - skipping summarization demo")
            return
        
        print("\n" + "="*60)
        print("📝 DEMO: Document Summarization")
        print("="*60)
        
        if not self.indexer.indexed_documents:
            print("⚠️  No documents indexed - skipping summarization demo")
            return
        
        # Get a sample document
        sample_doc = self.indexer.indexed_documents[0]
        source = Path(sample_doc.metadata.get('source', 'Unknown')).name
        
        print(f"📄 Summarizing document: {source}")
        
        # Try different summary styles
        styles = [
            (SummaryStyle.TLDR, SummaryLength.SHORT),
            (SummaryStyle.BULLETS, SummaryLength.MEDIUM),
            (SummaryStyle.STRUCTURED, SummaryLength.MEDIUM)
        ]
        
        for style, length in styles:
            try:
                print(f"\n📋 {style.value.upper()} Summary ({length.value}):")
                
                result = await self.summary_manager.summarizer.summarize_document(
                    sample_doc, style, length
                )
                
                print(f"✨ {result.summary}")
                print(f"🔢 Tokens used: {result.token_count}")
                
            except Exception as e:
                print(f"❌ Error generating {style.value} summary: {e}")
        
        # Demonstrate search result summarization
        print(f"\n📋 Search Results Summary:")
        try:
            search_results = self.hybrid_searcher.search("financial performance", top_k=3)
            
            if search_results:
                result = await self.summary_manager.summarizer.summarize_search_results(
                    search_results, 
                    "financial performance",
                    SummaryStyle.STRUCTURED,
                    SummaryLength.MEDIUM
                )
                
                print(f"✨ {result.summary}")
                print(f"🔢 Tokens used: {result.token_count}")
                print(f"📚 Sources: {', '.join([Path(s).name for s in result.sources])}")
        
        except Exception as e:
            print(f"❌ Error generating search summary: {e}")
    
    def demo_statistics(self):
        """Show system statistics and capabilities."""
        print("\n" + "="*60)
        print("📊 DEMO: System Statistics")
        print("="*60)
        
        # Index statistics
        index_stats = self.indexer.get_index_stats()
        print("📚 Index Statistics:")
        for key, value in index_stats.items():
            print(f"   {key}: {value}")
        
        # Hybrid search statistics
        if hasattr(self.hybrid_searcher, 'get_search_stats'):
            search_stats = self.hybrid_searcher.get_search_stats()
            print(f"\n🔍 Search Statistics:")
            for key, value in search_stats.items():
                print(f"   {key}: {value}")
        
        # LLM information
        if self.llm_router:
            provider_info = self.llm_router.get_provider_info()
            print(f"\n🤖 LLM Information:")
            for key, value in provider_info.items():
                if key == "available_providers":
                    print(f"   {key}: {', '.join(value)}")
                else:
                    print(f"   {key}: {value}")
        
        # Available providers
        available_providers = LLMRouter.get_available_providers()
        print(f"\n🔌 Available LLM Providers:")
        for provider, info in available_providers.items():
            print(f"   {provider}: {', '.join(info['models'])}")


async def main():
    """Main demo function."""
    print("🎯 Enhanced Semantic Document Search Engine - Complete Demo")
    print("=" * 70)
    
    # Initialize engine
    engine = EnhancedSemanticEngine()
    
    # Try to setup LLM (will gracefully handle missing API keys)
    llm_setup_success = engine.setup_llm()
    
    # Build index
    index_success = engine.build_index()
    
    if not index_success:
        print("\n❌ Cannot proceed without indexed documents")
        return
    
    # Run all demos
    print("\n🎪 Running demonstration of all features...")
    
    # Basic demos (work without LLM)
    engine.demo_basic_search()
    engine.demo_hybrid_search()
    engine.demo_statistics()
    
    # Advanced demos (require LLM)
    if llm_setup_success:
        await engine.demo_conversation()
        await engine.demo_summarization()
    else:
        print("\n💡 To enable AI features, set up your API keys:")
        print("   - OPENAI_API_KEY for OpenAI")
        print("   - ANTHROPIC_API_KEY for Claude")
        print("   - GOOGLE_API_KEY for Gemini")
        print("   - XAI_API_KEY for xAI (future)")
    
    print("\n🎉 Demo completed! All features showcased.")


def interactive_mode():
    """Interactive mode for testing features."""
    print("\n🎮 Interactive Mode")
    print("Commands:")
    print("  1 - Basic search demo")
    print("  2 - Hybrid search demo")
    print("  3 - Conversation demo") 
    print("  4 - Summarization demo")
    print("  5 - System statistics")
    print("  q - Quit")
    
    engine = EnhancedSemanticEngine()
    engine.setup_llm()
    engine.build_index()
    
    while True:
        try:
            choice = input("\nSelect option> ").strip().lower()
            
            if choice == 'q':
                print("Goodbye!")
                break
            elif choice == '1':
                engine.demo_basic_search()
            elif choice == '2':
                engine.demo_hybrid_search()
            elif choice == '3':
                asyncio.run(engine.demo_conversation())
            elif choice == '4':
                asyncio.run(engine.demo_summarization())
            elif choice == '5':
                engine.demo_statistics()
            else:
                print("Invalid option. Try again.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        asyncio.run(main())