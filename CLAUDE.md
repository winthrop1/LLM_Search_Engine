# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an enhanced semantic PDF search engine that combines semantic search with keyword matching, supports multiple document formats, includes OCR capabilities, and provides conversational AI features. The system uses FAISS for vector storage, LangChain for document processing, and supports multiple LLM providers (OpenAI, Claude, Gemini, xAI).

## Core Architecture

### Key Components

- **`src/search.py`**: HybridSearcher combining semantic and keyword search with TF-IDF scoring
- **`src/indexing.py`**: IncrementalIndexer with document tracking and selective reprocessing
- **`src/ingestion.py`**: Multi-format document loader (PDF, DOCX, TXT, HTML, PPTX, XLSX)
- **`src/ocr.py`**: OCR fallback for scanned documents using Tesseract
- **`src/llm_router.py`**: Unified LLM provider interface supporting streaming
- **`src/conversation.py`**: ConversationManager with context memory
- **`src/summarization.py`**: Document summarization with multiple styles
- **`src/preprocessing.py`**: Text cleaning and normalization pipeline

### Data Flow

1. Documents ingested from `./data/` folder
2. Text extracted and preprocessed 
3. Embeddings generated and stored in `./index/` (FAISS)
4. Metadata tracked in `./processed/document_metadata.json`
5. Search combines semantic similarity with keyword matching
6. Results can be summarized or used in conversations

## Development Commands

### Running the Application

```bash
# Enhanced version with all features
python main.py

# Full demo with interactive mode
python demo_enhanced.py
python demo_enhanced.py --interactive
```

### Testing

```bash
# Test multi-format support and features
python test_features.py

# Test search-only functionality
python test_search_only.py

# Test environment configuration
python test_env_config.py
```

### Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Install Tesseract OCR (required for scanned document support)
# macOS:
brew install tesseract

# Ubuntu/Debian:
sudo apt-get install tesseract-ocr
```

## Environment Configuration

Required `.env` file in project root:

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
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
XAI_API_KEY=your_key_here
```

## Interactive Commands

When running the enhanced applications, these commands are available:

- `/info` - Show index and system information
- `/update` - Update index incrementally  
- `/rebuild` - Force complete rebuild of index
- `/hybrid` - Use hybrid search (semantic + keyword)
- `/ai` - Ask AI question with conversation memory
- `/stream` - Toggle streaming responses on/off
- `/conv` - Show conversation information
- `/quit` - Exit application

## Key Design Patterns

### Hybrid Search
The system combines semantic embeddings with TF-IDF keyword scoring. Weights are configurable (default: 70% semantic, 30% keyword).

### Incremental Indexing
Documents are tracked by file hash to avoid reprocessing unchanged files. Only new or modified documents are reindexed.

### LLM Provider Abstraction
The LLMRouter provides a unified interface across providers with consistent streaming and response formatting.

### Conversation Memory
ConversationManager maintains context across multiple turns with configurable memory limits and token usage tracking.

## File Structure Notes

- `main.py` - Full-featured application entry point
- `demo_enhanced.py` - Demo script with examples
- `./data/` - Document storage directory
- `./index/` - FAISS vector store and embeddings
- `./processed/` - Metadata and conversation memory
- `test_*.py` - Feature testing scripts