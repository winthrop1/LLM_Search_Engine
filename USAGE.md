# Enhanced Semantic Document Search Engine - Usage Guide

This document provides comprehensive instructions for using the Enhanced Semantic Document Search Engine with all its advanced features.

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install Tesseract OCR (for document scanning support)
# On macOS:
brew install tesseract

# On Ubuntu/Debian:
sudo apt-get install tesseract-ocr

# On Windows:
# Download from https://github.com/UB-Mannheim/tesseract/wiki
```

### 2. Configuration

Create a `.env` file in the project root:

```env
# Document settings
DATA_FOLDER=./data
VECTOR_DB_NAME=sentence-transformers/all-MiniLM-L6-v2

# LLM Provider Configuration (choose one or more)
LLM_PROVIDER=openai  # options: openai, claude, gemini, xai
LLM_MODEL=gpt-3.5-turbo
MAX_TOKENS=1000      # Maximum tokens for LLM generation
USE_STREAMING=true   # Enable streaming responses (true/false)

# API Keys (add the ones you plan to use)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
XAI_API_KEY=your_xai_api_key_here
```

### 3. Running the Applications

#### Basic Enhanced Version
```bash
python main.py
```

#### Full Demo with All Features
```bash
python demo_enhanced.py

# For interactive mode:
python demo_enhanced.py --interactive
```

## Features Overview

### 🔍 Multi-Format Document Support
The engine supports various document formats:
- **PDF** (with OCR fallback for scanned documents)
- **Word Documents** (.docx)
- **Text Files** (.txt, .md)
- **HTML Files** (.html)
- **Presentations** (.pptx)
- **Spreadsheets** (.xlsx)

### 🤖 AI-Powered Features

#### Conversational Search
```python
# Example conversation flow
User: "What are the main financial metrics in the documents?"
AI: "Based on the documents, the main financial metrics include..."

User: "Can you provide more details about revenue?"
AI: "Building on the previous context, the revenue details show..."
```

#### Document Summarization
Available summary styles:
- **Plain**: Clear, straightforward summaries
- **TL;DR**: Quick essence summaries
- **Bullets**: Key points in bullet format
- **Structured**: Organized with clear sections
- **Executive**: Professional business format

#### Streaming Responses
Stream AI responses in real-time as they're generated:
- **Environment Control**: Set `USE_STREAMING=true` in `.env`
- **Interactive Toggle**: Use `/stream` command to toggle on/off
- **Visual Indicators**: Shows streaming progress with emojis and indicators
- **All Providers**: Works with OpenAI, Claude, Gemini, and xAI

## Detailed Usage Examples

### Basic Search

```bash
# Start the enhanced application
python main.py

# Interactive commands:
Search> financial performance
Search> /info          # Show index and system information
Search> /update        # Update index incrementally
Search> /rebuild       # Force complete rebuild
Search> /hybrid        # Use hybrid search (semantic + keyword)
Search> /ai            # Ask AI question with conversation memory
Search> /stream        # Toggle streaming responses on/off
Search> /conv          # Show conversation information
Search> /quit          # Exit application
```

### Hybrid Search (Semantic + Keyword)

```python
from src.search import HybridSearcher

# Initialize hybrid searcher
searcher = HybridSearcher(
    embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
    semantic_weight=0.7,  # 70% semantic, 30% keyword
    keyword_weight=0.3
)

# Index documents
searcher.index_documents(documents)

# Search with combined approach
results = searcher.search("revenue growth financial", top_k=5)

# Results include both semantic and keyword scores
for result in results:
    print(f"Combined Score: {result.combined_score}")
    print(f"Semantic: {result.semantic_score}")
    print(f"Keyword: {result.keyword_score}")
    print(f"Highlights: {result.match_highlights}")
```

### Conversational AI

```python
from src.conversation import ConversationManager
from src.llm_router import LLMRouter

# Setup LLM and conversation
llm_router = LLMRouter(provider="openai", model="gpt-3.5-turbo")
conv_manager = ConversationManager(llm_router, hybrid_searcher)

# Start conversation
conv_id = conv_manager.start_new_conversation()

# Ask questions with memory
response, results = await conv_manager.ask_question(
    "What are the key financial highlights?"
)

# Follow-up questions maintain context
response, results = await conv_manager.ask_question(
    "How do these compare to last year?"
)

# View conversation history
summary = conv_manager.get_conversation_summary()
print(f"Conversation turns: {summary['total_turns']}")
print(f"Total tokens used: {summary['total_tokens']}")
```

### Document Summarization

```python
from src.summarization import SummaryManager, SummaryStyle, SummaryLength

# Initialize summarizer
summary_manager = SummaryManager(llm_router)

# Summarize a single document
result = await summary_manager.summarizer.summarize_document(
    document, 
    style=SummaryStyle.BULLETS,
    length=SummaryLength.MEDIUM
)

print(result.summary)

# Summarize search results
search_results = hybrid_searcher.search("financial performance")
result = await summary_manager.summarizer.summarize_search_results(
    search_results,
    query="financial performance",
    style=SummaryStyle.STRUCTURED
)

print(result.summary)

# Smart summarization (auto-selects best style/length)
result = await summary_manager.smart_summarize(
    content=search_results,
    query="financial performance"
)
```

### LLM Provider Switching

```python
from src.llm_router import LLMRouter

# OpenAI
llm = LLMRouter(provider="openai", model="gpt-4")

# Claude
llm = LLMRouter(provider="claude", model="claude-3-sonnet-20240229")

# Gemini
llm = LLMRouter(provider="gemini", model="gemini-pro")

# xAI (future-compatible)
llm = LLMRouter(provider="xai", model="grok-1")

# Generate response
response = llm.generate("Summarize this document...")
print(f"Response: {response.output}")
print(f"Tokens: {response.tokens}")
print(f"Provider: {response.provider}")

# Streaming response
for chunk in llm.stream("Tell me about..."):
    print(chunk, end='', flush=True)
```

## Advanced Configuration

### Environment Variables

```env
# Core Settings
DATA_FOLDER=./data                    # Document directory
VECTOR_DB_NAME=sentence-transformers/all-MiniLM-L6-v2
VECTOR_TOKEN=your_hf_token           # Optional: for private models

# LLM Configuration
LLM_PROVIDER=openai                  # openai, claude, gemini, xai
LLM_MODEL=gpt-3.5-turbo             # Provider-specific model name
MAX_TOKENS=1000                      # Maximum tokens for generation
USE_STREAMING=true                   # Enable streaming responses

# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...
XAI_API_KEY=xai-...
```

### Programmatic Configuration

```python
# Custom hybrid search weights
hybrid_searcher = HybridSearcher(
    semantic_weight=0.8,  # Favor semantic similarity
    keyword_weight=0.2
)

# Conversation memory settings
conv_manager = ConversationManager(
    llm_router=llm_router,
    hybrid_searcher=hybrid_searcher,
    max_context_turns=10,      # Remember last 10 turns
    max_context_length=6000    # Limit context size
)

# Custom document processing
preprocessor = TextPreprocessor()
preprocessor.enable_hyphenation_fix = True
preprocessor.remove_headers_footers = True
preprocessor.normalize_whitespace = True
```

## Performance Tips

### Indexing
- Use incremental indexing (`/update`) for regular updates
- Force rebuild (`/rebuild`) only when necessary
- Monitor index statistics with `/info`

### Search
- Adjust hybrid search weights based on your use case
- Use appropriate `top_k` values (5-10 for most cases)
- Consider document types when setting search parameters

### LLM Usage
- Choose models based on task complexity:
  - `gpt-3.5-turbo` for general use
  - `gpt-4` for complex reasoning
  - `claude-3-sonnet` for long documents
  - `gemini-pro` for multimodal content

### Memory Management
- Monitor token usage in conversations
- Clear conversation history periodically
- Use shorter summary lengths for frequent operations

## Troubleshooting

### Common Issues

1. **No documents found**
   - Check `data/` path in `.env`
   - Ensure supported file formats are present
   - Verify file permissions

2. **OCR not working**
   - Install Tesseract: `brew install tesseract` (macOS)
   - Check Tesseract installation: `tesseract --version`

3. **LLM API errors**
   - Verify API keys are correctly set
   - Check API key permissions and quotas
   - Ensure internet connectivity

4. **Memory issues**
   - Reduce batch sizes for large document sets
   - Use incremental indexing instead of full rebuild
   - Monitor system memory usage

### Performance Optimization

```bash
# Monitor index performance
python -c "
from src.indexing import IncrementalIndexer
indexer = IncrementalIndexer('./index')
stats = indexer.get_index_stats()
print(stats)
"

# Check hybrid search statistics
python -c "
from src.search import HybridSearcher
searcher = HybridSearcher()
# ... after indexing documents
stats = searcher.get_search_stats()
print(stats)
"
```

## GitHub Integration

This project is designed for easy deployment and sharing on GitHub:

### Repository Setup
```bash
# Initialize git repository (if not already done)
git init
git add .
git commit -m "Initial commit: Enhanced Semantic Document Search Engine"

# Add remote repository
git remote add origin https://github.com/winthrop1/LLM_Search_Engine.git
git push -u origin main
```

### Environment Variables for GitHub Actions
For CI/CD pipelines, set these secrets in your GitHub repository:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY` 
- `GOOGLE_API_KEY`
- `XAI_API_KEY`

## API Reference

For detailed API documentation, see the docstrings in each module:
- `src/search.py` - Hybrid search functionality
- `src/llm_router.py` - LLM provider abstraction
- `src/conversation.py` - Conversational AI
- `src/summarization.py` - Document summarization
- `src/ingestion.py` - Multi-format document loading
- `src/preprocessing.py` - Text processing pipeline
- `src/indexing.py` - Incremental indexing system
