# Enhanced Semantic Document Search Engine

An advanced semantic search engine that combines semantic search with keyword matching, supports multiple document formats, includes OCR capabilities, and provides conversational AI features. The system uses FAISS for vector storage, LangChain for document processing, and supports multiple LLM providers (OpenAI, Claude, Gemini, xAI).

## Features

### 🔍 Advanced Search Capabilities
- **Hybrid Search**: Combines semantic embeddings with TF-IDF keyword matching
- **Multi-Format Support**: PDF, DOCX, TXT, MD, HTML, PPTX, XLSX
- **OCR Integration**: Automatic text extraction from scanned documents using Tesseract
- **Incremental Indexing**: Smart document tracking with selective reprocessing

### 🤖 AI-Powered Features
- **Conversational Search**: Multi-turn conversations with context memory
- **Document Summarization**: Multiple styles (Plain, TL;DR, Bullets, Structured, Executive)
- **Streaming Responses**: Real-time response generation
- **Multi-LLM Support**: OpenAI, Claude, Gemini, and xAI integration

### 📚 Document Processing
- **Advanced Text Processing**: Header/footer removal, hyphenation correction
- **Smart Preprocessing**: Whitespace normalization, content structure preservation
- **Metadata Tracking**: Comprehensive document metadata and change detection

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/winthrop1/LLM_Search_Engine
cd semantic_llm

# Install Python dependencies
pip install -r requirements.txt

# Install Tesseract OCR
# macOS:
brew install tesseract

# Ubuntu/Debian:
sudo apt-get install tesseract-ocr
```

### 2. Configuration

Create a `.env` file in the project root:

```env
# Document settings
DATA_FOLDER=./data
VECTOR_DB_NAME=sentence-transformers/all-MiniLM-L6-v2

# LLM Provider (choose one)
LLM_PROVIDER=openai  # openai, claude, gemini, xai
LLM_MODEL=gpt-3.5-turbo
MAX_TOKENS=1000
USE_STREAMING=true

# API Keys (add the ones you plan to use)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
XAI_API_KEY=your_xai_api_key_here
```

### 3. Usage

```bash
# Run the full demo with examples
python demo_enhanced.py

# Interactive demo mode
python demo_enhanced.py --interactive

# Run the enhanced search engine
python main.py
```

## Interactive Commands

When running the application, use these commands:

- `/info` - Show index and system information
- `/update` - Update index incrementally  
- `/rebuild` - Force complete rebuild of index
- `/hybrid` - Use hybrid search (semantic + keyword)
- `/ai` - Ask AI question with conversation memory
- `/stream` - Toggle streaming responses on/off
- `/conv` - Show conversation information
- `/quit` - Exit application

## Project Structure

```
./
├── data/              # Document storage directory
├── index/             # FAISS vector store and embeddings
├── processed/         # Document metadata and conversation memory
├── src/               # Source code modules
│   ├── conversation.py    # Conversational AI with memory
│   ├── indexing.py       # Incremental indexing system
│   ├── ingestion.py      # Multi-format document loading
│   ├── llm_router.py     # LLM provider abstraction
│   ├── ocr.py           # OCR processing
│   ├── preprocessing.py  # Text processing pipeline
│   ├── search.py        # Hybrid search functionality
│   └── summarization.py # Document summarization
├── main.py            # Main application entry point
├── demo_enhanced.py   # Feature demonstration script
└── test_*.py         # Testing scripts
```

## Documentation

- **[USAGE.md](USAGE.md)** - Comprehensive usage guide with examples

## Testing

```bash
# Test multi-format support and features
python test_features.py

# Test search-only functionality
python test_search_only.py

# Test environment configuration
python test_env_config.py
```

## Architecture Highlights

### Hybrid Search System
Combines semantic embeddings (70%) with TF-IDF keyword matching (30%) for optimal search results across different query types.

### Incremental Indexing
Uses SHA-256 file hashing to track document changes, ensuring only modified files are reprocessed.

### LLM Provider Abstraction
Unified interface across OpenAI, Claude, Gemini, and xAI with consistent streaming and response formatting.

### Conversation Memory
Maintains context across multiple turns with configurable memory limits and intelligent context building.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is open source. Please see the license file for details.
