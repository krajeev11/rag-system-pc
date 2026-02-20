"""
Vector Store Module using ChromaDB
Stores and retrieves document embeddings using ChromaDB
"""

from typing import List, Dict, Tuple, Optional
import os
from pathlib import Path


class ChromaVectorStore:
    """Manages vector storage and similarity search using ChromaDB"""
    
    def __init__(
        self, 
        collection_name: str = "rag_documents",
        persist_directory: Optional[str] = None,
        embedding_function=None
    ):
        """
        Initialize the Chroma vector store
        
        Args:
            collection_name: Name of the Chroma collection
            persist_directory: Directory to persist the database (None for in-memory)
            embedding_function: LangChain embedding function (required)
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self.vectorstore = None
        self._initialize_vectorstore()
    
    def _initialize_vectorstore(self):
        """Initialize the Chroma vector store"""
        try:
            from langchain_community.vectorstores import Chroma
            
            if self.embedding_function is None:
                raise ValueError("embedding_function is required for ChromaVectorStore")
            
            # Initialize Chroma vectorstore
            if self.persist_directory:
                # Persistent storage
                self.vectorstore = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embedding_function,
                    persist_directory=self.persist_directory
                )
            else:
                # In-memory storage
                self.vectorstore = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embedding_function
                )
        except ImportError as e:
            raise ImportError(
                f"ChromaDB dependencies not installed. Install with: pip install chromadb langchain-community. Error: {e}"
            )
    
    def add_documents(
        self,
        documents: List,
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ):
        """
        Add documents to the vector store
        
        Args:
            documents: List of LangChain Document objects or text strings
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of document IDs
        """
        if self.vectorstore is None:
            self._initialize_vectorstore()
        
        # If documents are strings, convert to LangChain Documents
        from langchain.schema import Document
        
        if documents and isinstance(documents[0], str):
            if metadatas is None:
                metadatas = [{}] * len(documents)
            documents = [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(documents, metadatas)
            ]
        
        # Add to Chroma
        if ids:
            self.vectorstore.add_documents(documents=documents, ids=ids)
        else:
            self.vectorstore.add_documents(documents=documents)
        
        # Persist if using persistent storage
        if self.persist_directory:
            self.vectorstore.persist()
    
    def add_embeddings(
        self,
        embeddings,
        chunks: List[Dict[str, str]],
        metadata: Optional[List[Dict]] = None
    ):
        """
        Add embeddings and associated chunks to the vector store
        This method maintains compatibility with the old interface
        
        Args:
            embeddings: Numpy array of embeddings (not used directly with Chroma)
            chunks: List of chunk dictionaries with 'content' and 'source' keys
            metadata: Optional list of metadata dictionaries
        """
        from langchain.schema import Document
        
        # Convert chunks to LangChain Documents
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            content = chunk.get('content', '')
            source = chunk.get('source', 'Unknown')
            chunk_idx = chunk.get('chunk_index', i)
            doc_idx = chunk.get('document_index', 0)
            
            doc_metadata = {
                'source': source,
                'chunk_index': chunk_idx,
                'document_index': doc_idx
            }
            
            if metadata and i < len(metadata):
                doc_metadata.update(metadata[i])
            
            documents.append(Document(page_content=content, metadata=doc_metadata))
            metadatas.append(doc_metadata)
        
        self.add_documents(documents, metadatas=metadatas)
    
    def search(
        self,
        query_embedding,
        k: int = 5,
        query_text: Optional[str] = None
    ) -> List[Tuple[Dict[str, str], float, Dict]]:
        """
        Search for similar chunks
        
        Args:
            query_embedding: Query embedding vector (not used directly, query_text preferred)
            query_text: Query text string (preferred method)
            k: Number of results to return
            
        Returns:
            List of tuples: (chunk_dict, similarity_score, metadata_dict)
        """
        if self.vectorstore is None:
            return []
        
        if query_text is None:
            raise ValueError("query_text is required for Chroma search")
        
        # Use similarity_search_with_score for better results
        results = self.vectorstore.similarity_search_with_score(query_text, k=k)
        
        formatted_results = []
        for doc, score in results:
            chunk_dict = {
                'content': doc.page_content,
                'source': doc.metadata.get('source', 'Unknown')
            }
            # Chroma returns distance (lower is better), convert to similarity
            similarity = 1 / (1 + score) if score > 0 else 1.0
            formatted_results.append((
                chunk_dict,
                float(similarity),
                doc.metadata
            ))
        
        return formatted_results
    
    def save(self, filepath: str):
        """
        Save the vector store to disk
        
        Args:
            filepath: Path to save the vector store
        """
        if self.vectorstore is None:
            raise ValueError("No vectorstore to save")
        
        # Create directory if it doesn't exist
        persist_dir = os.path.dirname(filepath) if os.path.dirname(filepath) else filepath
        os.makedirs(persist_dir, exist_ok=True)
        
        # Update persist directory and persist
        self.persist_directory = filepath
        self.vectorstore.persist()
        print(f"Vector store saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load the vector store from disk
        
        Args:
            filepath: Path to load the vector store from
        """
        if self.embedding_function is None:
            raise ValueError("embedding_function is required to load Chroma vectorstore")
        
        try:
            from langchain_community.vectorstores import Chroma
            
            self.persist_directory = filepath
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_function,
                persist_directory=filepath
            )
            print(f"Vector store loaded from {filepath}")
        except Exception as e:
            raise FileNotFoundError(f"Could not load vector store from {filepath}: {e}")
    
    def get_size(self) -> int:
        """Get the number of documents stored"""
        if self.vectorstore is None:
            return 0
        
        try:
            # Get collection count
            collection = self.vectorstore._collection
            return collection.count()
        except:
            return 0
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        if self.embedding_function is None:
            return 384  # Default for sentence-transformers
        
        # Try to get dimension from embedding function
        try:
            test_embedding = self.embedding_function.embed_query("test")
            return len(test_embedding)
        except:
            return 384
