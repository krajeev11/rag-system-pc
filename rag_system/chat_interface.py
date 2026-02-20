"""
Interactive Chat Interface for RAG System
Provides a command-line chat interface for querying documents
"""

import os
import sys
from typing import Optional
from datetime import datetime


class ChatInterface:
    """Interactive chat interface for the RAG system"""
    
    def __init__(self, rag_system):
        """
        Initialize the chat interface
        
        Args:
            rag_system: RAGSystem instance
        """
        self.rag = rag_system
        self.conversation_history = []
        self.running = False
    
    def print_welcome(self):
        """Print welcome message"""
        print("\n" + "=" * 70)
        print("🤖 RAG Chat Interface - Ask questions about your documents")
        print("=" * 70)
        print("\nCommands:")
        print("  /help     - Show this help message")
        print("  /stats    - Show system statistics")
        print("  /clear    - Clear conversation history")
        print("  /exit     - Exit the chat")
        print("\nType your questions below. Press Ctrl+C or type '/exit' to quit.")
        print("-" * 70)
    
    def print_stats(self):
        """Print system statistics"""
        stats = self.rag.get_stats()
        print("\n" + "=" * 70)
        print("System Statistics:")
        print("=" * 70)
        print(f"  Total chunks indexed: {stats['num_chunks']}")
        print(f"  Embedding dimension: {stats['embedding_dim']}")
        print(f"  Generator type: {stats['generator_type']}")
        print(f"  Conversation turns: {len(self.conversation_history)}")
        print("=" * 70 + "\n")
    
    def process_command(self, user_input: str) -> bool:
        """
        Process special commands
        
        Args:
            user_input: User input string
            
        Returns:
            True if command was processed, False otherwise
        """
        command = user_input.strip().lower()
        
        if command == '/help':
            self.print_welcome()
            return True
        elif command == '/stats':
            self.print_stats()
            return True
        elif command == '/clear':
            self.conversation_history = []
            print("\n✓ Conversation history cleared\n")
            return True
        elif command == '/exit' or command == '/quit':
            print("\n👋 Goodbye!")
            return False
        else:
            return None  # Not a command
    
    def format_response(self, result: dict, query: str) -> str:
        """Format the response for display"""
        output = []
        output.append(f"\n{'='*70}")
        output.append(f"Question: {query}")
        output.append(f"{'='*70}")
        output.append(f"\nAnswer:\n{result['answer']}")
        
        if result.get('sources'):
            output.append(f"\n📚 Sources ({len(result['sources'])}):")
            for i, source in enumerate(result['sources'][:5], 1):  # Show top 5 sources
                output.append(f"  {i}. {source}")
            if len(result['sources']) > 5:
                output.append(f"  ... and {len(result['sources']) - 5} more")
        
        if result.get('scores'):
            output.append(f"\n📊 Relevance scores: {[f'{s:.3f}' for s in result['scores'][:3]]}")
        
        output.append(f"\n{'-'*70}\n")
        return "\n".join(output)
    
    def chat(self, top_k: int = 5):
        """
        Start interactive chat session
        
        Args:
            top_k: Number of chunks to retrieve per query
        """
        self.print_welcome()
        self.running = True
        
        try:
            while self.running:
                try:
                    # Get user input
                    user_input = input("You: ").strip()
                    
                    if not user_input:
                        continue
                    
                    # Check for commands
                    command_result = self.process_command(user_input)
                    if command_result is False:
                        self.running = False
                        break
                    elif command_result is True:
                        continue
                    
                    # Process query
                    print("\n🔍 Searching documents...")
                    result = self.rag.query(user_input, top_k=top_k)
                    
                    # Display response
                    response = self.format_response(result, user_input)
                    print(response)
                    
                    # Save to history
                    self.conversation_history.append({
                        'query': user_input,
                        'answer': result['answer'],
                        'sources': result.get('sources', []),
                        'timestamp': datetime.now()
                    })
                
                except KeyboardInterrupt:
                    print("\n\n👋 Goodbye!")
                    self.running = False
                    break
                except Exception as e:
                    print(f"\n❌ Error: {e}")
                    print("Please try again or type '/exit' to quit.\n")
        
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
            sys.exit(1)
    
    def single_query(self, question: str, top_k: int = 5) -> dict:
        """
        Process a single query without interactive mode
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
            
        Returns:
            Query result dictionary
        """
        result = self.rag.query(question, top_k=top_k)
        return result


def create_chat_interface(rag_system):
    """Factory function to create a chat interface"""
    return ChatInterface(rag_system)
