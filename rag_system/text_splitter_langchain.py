"""
Text Splitter Module using LangChain
Splits documents into smaller chunks using LangChain's text splitters
"""

from typing import List, Dict


class LangChainTextSplitter:
    """Splits text into smaller chunks using LangChain's RecursiveCharacterTextSplitter"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the text splitter
        
        Args:
            chunk_size: Maximum size of each chunk (in characters)
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = None
        self._initialize_splitter()
    
    def _initialize_splitter(self):
        """Initialize the LangChain text splitter"""
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
        except ImportError as e:
            raise ImportError(
                f"LangChain not installed. Install with: pip install langchain. Error: {e}"
            )
    
    def split_text(self, text: str) -> List[str]:
        """
        Split a single text into chunks
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if self.text_splitter is None:
            self._initialize_splitter()
        
        chunks = self.text_splitter.split_text(text)
        return chunks
    
    def split_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Split multiple documents into chunks
        
        Args:
            documents: List of document dictionaries with 'content' and 'source' keys
            
        Returns:
            List of chunk dictionaries with 'content', 'source', and 'chunk_index' keys
        """
        if self.text_splitter is None:
            self._initialize_splitter()
        
        from langchain.schema import Document
        
        # Convert to LangChain Documents
        langchain_docs = [
            Document(page_content=doc['content'], metadata={'source': doc['source']})
            for doc in documents
        ]
        
        # Split using LangChain
        split_docs = self.text_splitter.split_documents(langchain_docs)
        
        # Convert back to our format
        all_chunks = []
        doc_chunk_counts = {}
        
        for doc in split_docs:
            source = doc.metadata.get('source', 'Unknown')
            
            # Track chunk index per document
            if source not in doc_chunk_counts:
                doc_chunk_counts[source] = 0
            
            all_chunks.append({
                'content': doc.page_content,
                'source': source,
                'chunk_index': doc_chunk_counts[source],
                'document_index': 0  # Can be enhanced to track document index
            })
            
            doc_chunk_counts[source] += 1
        
        return all_chunks
    
    def split_langchain_documents(self, documents: List) -> List:
        """
        Split LangChain Document objects
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            List of split LangChain Document objects
        """
        if self.text_splitter is None:
            self._initialize_splitter()
        
        return self.text_splitter.split_documents(documents)
