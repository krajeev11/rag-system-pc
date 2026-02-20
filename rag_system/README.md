# RAG System - Retrieval-Augmented Generation

A complete Python implementation of a Retrieval-Augmented Generation (RAG) system using **LangChain** and **ChromaDB**. Combines document retrieval with language generation. **Now with system-wide document scanning and interactive chat interface!**

## Features

- **Built on LangChain & ChromaDB**: Industry-standard RAG framework
- **System Document Scanning**: Automatically scan your Ubuntu/Linux system for documents
- **Interactive Chat Interface**: Chat with your documents using a command-line interface
- **Document Loading**: Load documents using LangChain's document loaders
- **Text Chunking**: Intelligent text splitting using LangChain's RecursiveCharacterTextSplitter
- **Embedding Generation**: Uses LangChain embeddings (sentence-transformers, OpenAI, etc.)
- **Vector Storage**: ChromaDB for efficient similarity search and persistence
- **Retrieval**: Semantic search to find relevant document chunks
- **Generation**: Template-based or LangChain LLM-powered response generation
- **Persistent Indexing**: Save and load vector stores for faster access

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

The core dependencies are:
- `langchain` - Core LangChain framework
- `langchain-community` - Community integrations for LangChain
- `chromadb` - Vector database for storing embeddings
- `sentence-transformers` - For generating embeddings (via LangChain)
- `numpy` - For numerical operations

Optional dependencies (for LLM integration):
- `openai` - For OpenAI API integration
- `anthropic` - For Anthropic Claude API integration

## Quick Start

### Basic Usage

```python
from rag_system import RAGSystem

# Initialize the RAG system
rag = RAGSystem(
    chunk_size=500,
    chunk_overlap=50,
    generator_type="simple"  # Use simple template-based generation
)

# Add documents
documents = [
    {
        'content': 'Your document content here...',
        'source': 'document1.txt'
    },
    # Add more documents...
]

rag.add_documents(documents)

# Query the system
result = rag.query("What is the main topic?")
print(result['answer'])
print(f"Sources: {result['sources']}")
```

### System Scanning

```python
from rag_system import RAGSystem

rag = RAGSystem()

# Scan your system for documents
rag.scan_and_index_system()

# Or scan specific directories
rag.scan_and_index_system(
    scan_directories=['~/Documents', '~/Downloads'],
    max_file_size_mb=100
)
```

### Loading from Files

```python
# Load from a single file
rag.add_documents("path/to/document.txt")

# Load from a directory
rag.add_documents("path/to/documents/")

# Load from multiple files
rag.add_documents(["file1.txt", "file2.txt", "file3.txt"])
```

### Using OpenAI API

```python
rag = RAGSystem(
    generator_type="openai",
    generator_api_key="your-api-key",
    generator_model="gpt-3.5-turbo"
)

rag.add_documents(documents)
result = rag.query("Your question here")
```

### Save and Load Vector Store

```python
# Save the vector store
rag.save("my_vector_store")

# Load in a new session
rag2 = RAGSystem()
rag2.load("my_vector_store")
result = rag2.query("Your question")
```

## Architecture

The RAG system consists of several components:

1. **DocumentLoader**: Handles loading documents from various sources
2. **TextSplitter**: Splits documents into smaller chunks
3. **Embedder**: Generates vector embeddings using sentence transformers
4. **VectorStore**: Stores embeddings and performs similarity search using FAISS
5. **Generator**: Generates responses using retrieved context
6. **RAGSystem**: Main class that orchestrates all components

## Configuration Options

### RAGSystem Parameters

- `chunk_size` (int): Size of text chunks (default: 500)
- `chunk_overlap` (int): Overlap between chunks (default: 50)
- `embedding_model` (str): Sentence transformer model name (default: "all-MiniLM-L6-v2")
- `generator_type` (str): Type of generator - "simple", "openai", "anthropic", or "local"
- `generator_api_key` (str): API key for generator (if required)
- `generator_model` (str): Model name for generator

### Query Parameters

- `question` (str): User question
- `top_k` (int): Number of relevant chunks to retrieve (default: 5)
- `max_tokens` (int): Maximum tokens for generation (default: 500)
- `temperature` (float): Sampling temperature (default: 0.7)

## Command-Line Interface

The `rag_chat.py` script provides a full-featured CLI:

```bash
# Basic usage - scan and chat
python rag_chat.py --scan

# Options:
#   --scan              Scan system for documents
#   --scan-dir DIR      Specific directories to scan
#   --load-index PATH   Load existing index
#   --save-index PATH   Save index path (default: rag_index)
#   --generator TYPE    Generator type: simple, openai, anthropic
#   --api-key KEY       API key for OpenAI/Anthropic
#   --top-k N           Number of chunks to retrieve (default: 5)
#   --chunk-size N      Text chunk size (default: 500)
```

### Chat Commands

Once in chat mode, you can use these commands:
- `/help` - Show help message
- `/stats` - Show system statistics
- `/clear` - Clear conversation history
- `/exit` - Exit the chat

## Examples

### Example 1: Scan System and Chat

```bash
python rag_chat.py --scan
```

This will:
1. Scan common user directories (Documents, Downloads, Desktop, etc.)
2. Index all found documents
3. Start an interactive chat interface

### Example 2: Scan Specific Directory

```bash
python rag_chat.py --scan --scan-dir ~/Documents/Projects
```

### Example 3: Load Existing Index

```bash
python rag_chat.py --load-index rag_index
```

### Example 4: Python API Usage

See `example.py` for complete usage examples:

```bash
python rag_system/example.py
```

## Supported Document Formats

The system scanner automatically finds and indexes:
- `.txt` - Plain text files
- `.md` - Markdown files
- `.py` - Python files
- `.js` - JavaScript files
- `.html` - HTML files
- `.css` - CSS files
- `.json` - JSON files
- `.xml` - XML files
- `.csv` - CSV files
- `.log` - Log files
- `.conf`, `.cfg`, `.ini` - Configuration files
- `.yaml`, `.yml` - YAML files
- `.rst` - reStructuredText files
- `.tex` - LaTeX files
- `.org` - Org-mode files
- `.wiki` - Wiki files
- `.rtf` - Rich Text Format files

## System Scanning

The document scanner automatically:
- Scans common user directories (Documents, Downloads, Desktop, etc.)
- Excludes system directories (proc, sys, dev, etc.)
- Skips hidden files and directories (except common ones like .bashrc)
- Respects file size limits (default: 50MB max per file)
- Provides progress updates during scanning

### Customizing Scan Directories

```python
from rag_system import RAGSystem

rag = RAGSystem()

# Scan specific directories
rag.scan_and_index_system(
    scan_directories=['/path/to/docs', '/another/path'],
    exclude_directories=['/path/to/exclude'],
    max_file_size_mb=100
)
```

## Embedding Models

Default model: `all-MiniLM-L6-v2` (lightweight, fast)

Alternative models (better quality, slower):
- `all-mpnet-base-v2` - Better quality embeddings
- `paraphrase-multilingual-MiniLM-L12-v2` - Multilingual support

## Limitations

- The simple generator uses template-based responses (no LLM)
- For better generation quality, use OpenAI or Anthropic API
- Vector store is stored in memory by default (use save/load for persistence)
- System scanning may take time for large directory structures
- Large files (>50MB default) are skipped during scanning

## Future Enhancements

- Support for more document formats (PDF, DOCX, etc.)
- Advanced chunking strategies (semantic chunking)
- Hybrid search (keyword + semantic)
- Local LLM integration (using transformers)
- Web UI for querying
- Incremental indexing (add new documents without full rescan)
- Document filtering and search refinement

