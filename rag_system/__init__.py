"""
RAG System - Retrieval-Augmented Generation Implementation using LangChain and ChromaDB
"""

# Import LangChain-based implementation
from .rag_langchain import RAGSystem
from .document_loader_langchain import LangChainDocumentLoader as DocumentLoader
from .text_splitter_langchain import LangChainTextSplitter as TextSplitter
from .embedder_langchain import LangChainEmbedder as Embedder
from .vector_store_chroma import ChromaVectorStore as VectorStore
from .generator_langchain import LangChainGenerator as Generator
from .document_scanner import DocumentScanner
from .chat_interface import ChatInterface

# Web UI interfaces
try:
    from .gradio_interface import GradioChatInterface
except ImportError:
    GradioChatInterface = None

try:
    from .streamlit_interface import StreamlitChatInterface
except ImportError:
    StreamlitChatInterface = None

__all__ = [
    'RAGSystem',
    'DocumentLoader',
    'TextSplitter',
    'Embedder',
    'VectorStore',
    'Generator',
    'DocumentScanner',
    'ChatInterface',
    'GradioChatInterface',
    'StreamlitChatInterface'
]

__version__ = '2.0.0'
