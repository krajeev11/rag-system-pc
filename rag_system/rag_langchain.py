"""
Main RAG System using LangChain and ChromaDB
Combines document loading, embedding, vector storage, retrieval, and generation
"""

from typing import List, Dict, Optional, Union
import os

from .document_loader_langchain import LangChainDocumentLoader
from .text_splitter_langchain import LangChainTextSplitter
from .embedder_langchain import LangChainEmbedder
from .vector_store_chroma import ChromaVectorStore
from .generator_langchain import LangChainGenerator
from .document_scanner import DocumentScanner


class RAGSystem:
    """
    Complete RAG System using LangChain and ChromaDB:
    1. Document loading and preprocessing (LangChain)
    2. Text chunking (LangChain)
    3. Embedding generation (LangChain)
    4. Vector storage (ChromaDB)
    5. Retrieval (ChromaDB)
    6. Response generation (LangChain LLMs)
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        generator_type: str = "simple",
        generator_api_key: Optional[str] = None,
        generator_model: str = "gpt-3.5-turbo",
        collection_name: str = "rag_documents",
        persist_directory: Optional[str] = None
    ):
        """
        Initialize the RAG system with LangChain and Chroma
        
        Args:
            chunk_size: Size of text chunks for embedding
            chunk_overlap: Overlap between chunks
            embedding_model: Embedding model name (sentence-transformers model or "openai")
            generator_type: Type of generator ("simple", "openai", "anthropic", "local")
            generator_api_key: API key for generator (if needed)
            generator_model: Model name for generator
            collection_name: Name of Chroma collection
            persist_directory: Directory to persist Chroma database (None for in-memory)
        """
        # Initialize LangChain components
        self.document_loader = LangChainDocumentLoader()
        self.text_splitter = LangChainTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Initialize embedder
        self.embedder = LangChainEmbedder(embedding_model=embedding_model)
        embeddings = self.embedder.get_langchain_embeddings()
        
        # Initialize Chroma vector store
        self.vector_store = ChromaVectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        
        # Initialize generator
        if generator_type == "simple":
            self.generator = None  # Will use simple template-based generation
        else:
            self.generator = LangChainGenerator(
                model_type=generator_type,
                api_key=generator_api_key,
                model_name=generator_model,
                vectorstore=self.vector_store.vectorstore
            )
        
        self.generator_type = generator_type
        self.document_scanner = None  # Will be initialized when needed
        self.persist_directory = persist_directory
    
    def add_documents(
        self,
        documents: Union[str, List[str], List[Dict[str, str]]],
        sources: Optional[List[str]] = None
    ):
        """
        Add documents to the RAG system
        
        Args:
            documents: Can be:
                      - File path (str)
                      - Directory path (str)
                      - List of file paths
                      - List of text strings
                      - List of document dictionaries
            sources: Optional list of source identifiers (if documents is list of strings)
        """
        # Load documents based on input type
        if isinstance(documents, str):
            # Check if it's a file or directory
            if os.path.isfile(documents):
                docs = [self.document_loader.load_from_file(documents)]
            elif os.path.isdir(documents):
                docs = self.document_loader.load_from_directory(documents)
            else:
                raise ValueError(f"Path does not exist: {documents}")
        elif isinstance(documents, list):
            if len(documents) == 0:
                return
            
            # Check if it's a list of file paths or text strings
            if isinstance(documents[0], str):
                # Check if first item is a file path
                if os.path.exists(documents[0]):
                    # List of file paths - use LangChain loader for efficiency
                    langchain_docs = self.document_loader.load_langchain_documents(documents)
                    # Split documents
                    split_docs = self.text_splitter.split_langchain_documents(langchain_docs)
                    # Add directly to vector store
                    self.vector_store.add_documents(split_docs)
                    print(f"Added {len(split_docs)} chunks from {len(documents)} files to the vector store")
                    return
                else:
                    # List of text strings
                    docs = self.document_loader.load_from_list(documents, sources)
            elif isinstance(documents[0], dict):
                # List of document dictionaries
                docs = documents
            else:
                raise ValueError("Unsupported document format")
        else:
            raise ValueError("Documents must be a string, list of strings, or list of dictionaries")
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(docs)
        
        # Convert to LangChain Documents and add to vector store
        from langchain.schema import Document
        
        langchain_docs = []
        for chunk in chunks:
            langchain_docs.append(Document(
                page_content=chunk['content'],
                metadata={
                    'source': chunk.get('source', 'Unknown'),
                    'chunk_index': chunk.get('chunk_index', 0),
                    'document_index': chunk.get('document_index', 0)
                }
            ))
        
        self.vector_store.add_documents(langchain_docs)
        
        print(f"Added {len(chunks)} chunks from {len(docs)} documents to the vector store")
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> Dict[str, any]:
        """
        Query the RAG system
        
        Args:
            question: User question
            top_k: Number of relevant chunks to retrieve
            max_tokens: Maximum tokens for generation
            temperature: Sampling temperature
            
        Returns:
            Dictionary with 'answer', 'sources', and 'chunks' keys
        """
        # Retrieve relevant chunks using Chroma
        results = self.vector_store.search(
            query_embedding=None,  # Not used with Chroma
            query_text=question,
            k=top_k
        )
        
        if not results:
            return {
                'answer': "I don't have enough information to answer this question.",
                'sources': [],
                'chunks': []
            }
        
        # Extract chunks and sources
        chunks = [result[0] for result in results]
        sources = list(set([chunk['source'] for chunk in chunks]))
        
        # Generate answer
        if self.generator:
            answer = self.generator.generate(
                question,
                chunks,
                max_tokens=max_tokens,
                temperature=temperature
            )
        else:
            # Use simple template-based generation
            answer = self._simple_generate(question, chunks)
        
        return {
            'answer': answer,
            'sources': sources,
            'chunks': chunks,
            'scores': [result[1] for result in results]
        }
    
    def _simple_generate(self, question: str, chunks: List[Dict[str, str]]) -> str:
        """Simple template-based answer generation"""
        if not chunks:
            return f"I don't have enough information to answer: {question}"
        
        # Use the top chunk to generate a simple answer
        top_chunk = chunks[0]
        source = top_chunk.get('source', 'Unknown')
        content = top_chunk.get('content', '')
        
        answer = f"Based on the information from {source}:\n\n"
        answer += content
        
        # Add additional context if available
        if len(chunks) > 1:
            answer += f"\n\nAdditional relevant information:\n"
            for i, chunk in enumerate(chunks[1:3], 2):  # Include up to 2 more chunks
                answer += f"\n[{i}] {chunk.get('content', '')[:200]}...\n"
        
        return answer
    
    def save(self, filepath: str):
        """
        Save the vector store to disk
        
        Args:
            filepath: Path to save the vector store
        """
        self.vector_store.save(filepath)
        print(f"Vector store saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load the vector store from disk
        
        Args:
            filepath: Path to load the vector store from
        """
        # Reinitialize with persist directory
        embeddings = self.embedder.get_langchain_embeddings()
        self.vector_store = ChromaVectorStore(
            collection_name=self.vector_store.collection_name,
            persist_directory=filepath,
            embedding_function=embeddings
        )
        self.vector_store.load(filepath)
        print(f"Vector store loaded from {filepath}")
    
    def scan_and_index_system(
        self,
        scan_directories: Optional[List[str]] = None,
        exclude_directories: Optional[List[str]] = None,
        max_file_size_mb: int = 50,
        show_progress: bool = True
    ) -> dict:
        """
        Scan the system for documents and index them
        
        Args:
            scan_directories: Directories to scan (default: common user directories)
            exclude_directories: Directories to exclude
            max_file_size_mb: Maximum file size to process
            show_progress: Whether to show progress messages
            
        Returns:
            Dictionary with scan statistics
        """
        if self.document_scanner is None:
            self.document_scanner = DocumentScanner(
                scan_directories=scan_directories,
                exclude_directories=exclude_directories,
                max_file_size_mb=max_file_size_mb
            )
        
        # Scan for files
        if show_progress:
            print("\n🔍 Scanning system for documents...")
        
        files_found = self.document_scanner.scan_system(recursive=True)
        
        if not files_found:
            print("No documents found to index.")
            return {'files_found': 0, 'chunks_added': 0}
        
        # Get summary
        summary = self.document_scanner.get_file_summary(files_found)
        
        if show_progress:
            print(f"\n📊 Scan Summary:")
            print(f"  Total files: {summary['total_files']}")
            print(f"  Total size: {summary['total_size_mb']} MB")
            print(f"\n📁 Files by extension:")
            for ext, count in list(summary['by_extension'].items())[:10]:
                print(f"    {ext}: {count}")
        
        # Index the files
        if show_progress:
            print(f"\n📚 Indexing {len(files_found)} files...")
        
        chunks_before = self.vector_store.get_size()
        
        # Process files in batches to show progress
        batch_size = 50
        for i in range(0, len(files_found), batch_size):
            batch = files_found[i:i+batch_size]
            file_paths = [f['path'] for f in batch]
            
            try:
                self.add_documents(file_paths)
            except Exception as e:
                if show_progress:
                    print(f"  Warning: Could not index some files: {e}")
            
            if show_progress and (i + batch_size) % 100 == 0:
                print(f"  Indexed {min(i + batch_size, len(files_found))}/{len(files_found)} files...")
        
        chunks_after = self.vector_store.get_size()
        chunks_added = chunks_after - chunks_before
        
        if show_progress:
            print(f"\n✅ Indexing complete! Added {chunks_added} chunks from {len(files_found)} files.")
        
        return {
            'files_found': len(files_found),
            'chunks_added': chunks_added,
            'summary': summary
        }
    
    def get_stats(self) -> Dict[str, any]:
        """Get statistics about the RAG system"""
        return {
            'num_chunks': self.vector_store.get_size(),
            'embedding_dim': self.vector_store.get_embedding_dim(),
            'generator_type': self.generator_type,
            'vector_store': 'ChromaDB',
            'framework': 'LangChain'
        }
