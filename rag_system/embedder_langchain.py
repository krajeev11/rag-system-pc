"""
Embedding Module using LangChain
Generates vector embeddings for text chunks using LangChain's embedding models
"""

from typing import List
import numpy as np


class LangChainEmbedder:
    """Generates embeddings for text using LangChain's embedding models"""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedder
        
        Args:
            embedding_model: Name of the embedding model to use
                           Default: "sentence-transformers/all-MiniLM-L6-v2"
                           Alternatives: 
                           - "openai" (requires OpenAI API key)
                           - "huggingface" models
                           - Other LangChain supported embeddings
        """
        self.model_name = embedding_model
        self.embeddings = None
        self._load_model()
    
    def _load_model(self):
        """Load the LangChain embedding model"""
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            
            if self.model_name.startswith("sentence-transformers/"):
                # Use HuggingFace embeddings for sentence-transformers models
                model_name = self.model_name.replace("sentence-transformers/", "")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={'device': 'cpu'}
                )
            elif self.model_name == "openai":
                # Use OpenAI embeddings
                from langchain_openai import OpenAIEmbeddings
                import os
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable not set")
                self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            else:
                # Default to HuggingFace embeddings
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={'device': 'cpu'}
                )
        except ImportError as e:
            raise ImportError(
                f"LangChain dependencies not installed. Install with: pip install langchain langchain-community sentence-transformers. Error: {e}"
            )
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array representing the embedding vector
        """
        if self.embeddings is None:
            self._load_model()
        
        embedding = self.embeddings.embed_query(text)
        return np.array(embedding)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts (batch processing)
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Numpy array of shape (num_texts, embedding_dim)
        """
        if self.embeddings is None:
            self._load_model()
        
        embeddings = self.embeddings.embed_documents(texts)
        return np.array(embeddings)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors"""
        if self.embeddings is None:
            self._load_model()
        
        # Create a dummy embedding to get the dimension
        test_embedding = self.embed_text("test")
        return len(test_embedding)
    
    def get_langchain_embeddings(self):
        """Get the LangChain embeddings object for direct use"""
        if self.embeddings is None:
            self._load_model()
        return self.embeddings
