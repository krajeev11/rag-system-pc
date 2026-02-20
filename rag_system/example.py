"""
Example usage of the RAG System
"""

import sys
import os

# Add parent directory to path to allow importing rag_system
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem


def example_basic_usage():
    """Basic example of using the RAG system"""
    print("=" * 60)
    print("RAG System - Basic Usage Example")
    print("=" * 60)
    
    # Initialize RAG system
    rag = RAGSystem(
        chunk_size=300,
        chunk_overlap=50,
        generator_type="simple"  # Use simple template-based generation
    )
    
    # Add some sample documents
    sample_docs = [
        {
            'content': """
            Python is a high-level programming language known for its simplicity and readability.
            It was created by Guido van Rossum and first released in 1991. Python supports multiple
            programming paradigms including procedural, object-oriented, and functional programming.
            """,
            'source': 'python_intro.txt'
        },
        {
            'content': """
            Machine Learning is a subset of artificial intelligence that enables systems to learn
            and improve from experience without being explicitly programmed. Common ML algorithms
            include neural networks, decision trees, and support vector machines.
            """,
            'source': 'ml_intro.txt'
        },
        {
            'content': """
            RAG (Retrieval-Augmented Generation) combines information retrieval with language
            generation. It retrieves relevant documents from a knowledge base and uses them as
            context for generating accurate and informed responses.
            """,
            'source': 'rag_explanation.txt'
        }
    ]
    
    print("\n1. Adding documents to the RAG system...")
    rag.add_documents(sample_docs)
    
    print("\n2. Querying the RAG system...")
    questions = [
        "What is Python?",
        "Explain machine learning",
        "How does RAG work?"
    ]
    
    for question in questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")
        result = rag.query(question, top_k=2)
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources: {result['sources']}")
        print(f"Relevance scores: {[f'{s:.3f}' for s in result['scores']]}")
    
    # Get statistics
    print(f"\n{'='*60}")
    print("System Statistics:")
    print(f"{'='*60}")
    stats = rag.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")


def example_file_loading():
    """Example of loading documents from files"""
    print("\n" + "=" * 60)
    print("RAG System - File Loading Example")
    print("=" * 60)
    
    rag = RAGSystem(generator_type="simple")
    
    # Example: Load from a single file (if it exists)
    # rag.add_documents("path/to/document.txt")
    
    # Example: Load from a directory
    # rag.add_documents("path/to/documents/")
    
    print("File loading example (commented out - provide actual file paths)")


def example_with_openai():
    """Example using OpenAI API (requires API key)"""
    print("\n" + "=" * 60)
    print("RAG System - OpenAI Integration Example")
    print("=" * 60)
    
    # Uncomment and add your API key to use OpenAI
    # rag = RAGSystem(
    #     generator_type="openai",
    #     generator_api_key="your-api-key-here",
    #     generator_model="gpt-3.5-turbo"
    # )
    
    # rag.add_documents(["Your documents here"])
    # result = rag.query("Your question here")
    # print(result['answer'])
    
    print("OpenAI example (commented out - requires API key)")


def example_save_and_load():
    """Example of saving and loading the vector store"""
    print("\n" + "=" * 60)
    print("RAG System - Save/Load Example")
    print("=" * 60)
    
    rag = RAGSystem(generator_type="simple")
    
    # Add documents
    sample_docs = [
        {
            'content': "This is a sample document for testing save/load functionality.",
            'source': 'test_doc.txt'
        }
    ]
    rag.add_documents(sample_docs)
    
    # Save the vector store
    print("\nSaving vector store...")
    rag.save("rag_vector_store")
    
    # Create a new RAG instance and load
    print("\nLoading vector store in new instance...")
    rag2 = RAGSystem(generator_type="simple")
    rag2.load("rag_vector_store")
    
    # Query the loaded system
    result = rag2.query("What is in the test document?")
    print(f"\nAnswer: {result['answer']}")


if __name__ == "__main__":
    # Run examples
    example_basic_usage()
    example_file_loading()
    example_with_openai()
    example_save_and_load()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
