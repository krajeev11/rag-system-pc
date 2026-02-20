"""
Document Loader Module using LangChain
Handles loading documents from various sources using LangChain loaders
"""

import os
from typing import List, Dict
from pathlib import Path


class LangChainDocumentLoader:
    """Loads documents using LangChain's document loaders"""
    
    def __init__(self):
        """Initialize the document loader"""
        pass
    
    def load_from_file(self, file_path: str) -> Dict[str, str]:
        """
        Load text content from a single file using LangChain
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            Dictionary with 'content' and 'source' keys
        """
        try:
            from langchain_community.document_loaders import TextLoader
            
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
            
            if documents:
                return {
                    'content': documents[0].page_content,
                    'source': file_path
                }
            else:
                return {'content': '', 'source': file_path}
        except Exception as e:
            # Fallback to simple file reading
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return {'content': content, 'source': file_path}
            except Exception as e2:
                raise Exception(f"Error loading file {file_path}: {str(e2)}")
    
    def load_from_directory(self, directory_path: str, recursive: bool = True) -> List[Dict[str, str]]:
        """
        Load all supported files from a directory using LangChain
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to search recursively
            
        Returns:
            List of dictionaries with 'content' and 'source' keys
        """
        documents = []
        path = Path(directory_path)
        
        if not path.exists():
            raise ValueError(f"Directory {directory_path} does not exist")
        
        # Supported extensions
        supported_extensions = {'.txt', '.md', '.py', '.js', '.html', '.css', 
                               '.json', '.xml', '.csv', '.log', '.conf', '.cfg', 
                               '.ini', '.yaml', '.yml', '.rst', '.tex', '.org', 
                               '.wiki', '.rtf'}
        
        pattern = "**/*" if recursive else "*"
        
        for file_path in path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    doc = self.load_from_file(str(file_path))
                    if doc['content']:  # Only add non-empty documents
                        documents.append(doc)
                except Exception as e:
                    print(f"Warning: Could not load {file_path}: {e}")
        
        return documents
    
    def load_from_text(self, text: str, source: str = "manual_input") -> Dict[str, str]:
        """
        Create a document from raw text
        
        Args:
            text: Raw text content
            source: Source identifier for the text
            
        Returns:
            Dictionary with 'content' and 'source' keys
        """
        return {
            'content': text,
            'source': source
        }
    
    def load_from_list(self, texts: List[str], sources: List[str] = None) -> List[Dict[str, str]]:
        """
        Create documents from a list of texts
        
        Args:
            texts: List of text strings
            sources: Optional list of source identifiers
            
        Returns:
            List of dictionaries with 'content' and 'source' keys
        """
        if sources is None:
            sources = [f"text_{i}" for i in range(len(texts))]
        
        if len(texts) != len(sources):
            raise ValueError("Number of texts must match number of sources")
        
        return [
            {'content': text, 'source': source}
            for text, source in zip(texts, sources)
        ]
    
    def load_langchain_documents(self, file_paths: List[str]) -> List:
        """
        Load files as LangChain Document objects
        
        Args:
            file_paths: List of file paths
            
        Returns:
            List of LangChain Document objects
        """
        from langchain.schema import Document
        from langchain_community.document_loaders import TextLoader
        
        documents = []
        for file_path in file_paths:
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
        
        return documents
