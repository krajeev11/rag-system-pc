"""
Streamlit Web Interface for RAG System (Alternative to Gradio)
Provides a user-friendly web UI for querying documents
"""

import streamlit as st
from typing import Optional
from datetime import datetime


class StreamlitChatInterface:
    """Streamlit-based web interface for the RAG system"""
    
    def __init__(self, rag_system):
        """
        Initialize the Streamlit interface
        
        Args:
            rag_system: RAGSystem instance
        """
        self.rag = rag_system
        
        # Initialize session state
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        if 'stats' not in st.session_state:
            st.session_state.stats = self.rag.get_stats()
    
    def render(self):
        """Render the Streamlit interface"""
        st.set_page_config(
            page_title="RAG Chat Interface",
            page_icon="🤖",
            layout="wide"
        )
        
        # Header
        st.title("🤖 RAG Chat Interface")
        st.markdown("Ask questions about your documents using Retrieval-Augmented Generation.")
        
        # Sidebar
        with st.sidebar:
            st.header("📊 System Statistics")
            stats = st.session_state.stats
            st.metric("Total Chunks", stats['num_chunks'])
            st.metric("Embedding Dimension", stats['embedding_dim'])
            st.metric("Generator Type", stats['generator_type'])
            st.metric("Conversation Turns", len(st.session_state.conversation_history))
            
            st.divider()
            
            st.header("💡 Tips")
            st.markdown("""
            - Ask specific questions for better results
            - Questions are answered based on your indexed documents
            - Sources are shown below each answer
            - Use the clear button to reset conversation
            """)
            
            if st.button("🔄 Refresh Statistics"):
                st.session_state.stats = self.rag.get_stats()
                st.rerun()
            
            if st.button("🗑️ Clear Chat"):
                st.session_state.conversation_history = []
                st.rerun()
        
        # Main chat area
        st.header("💬 Chat")
        
        # Display conversation history
        for i, entry in enumerate(st.session_state.conversation_history):
            with st.chat_message("user"):
                st.write(entry['query'])
            
            with st.chat_message("assistant"):
                st.write(entry['answer'])
                
                # Show sources
                if entry.get('sources'):
                    with st.expander("📚 Sources"):
                        for j, source in enumerate(entry['sources'][:5], 1):
                            st.write(f"{j}. `{source}`")
        
        # Chat input
        user_input = st.chat_input("Type your question here...")
        
        if user_input:
            # Add user message
            st.session_state.conversation_history.append({
                'query': user_input,
                'answer': '',
                'sources': [],
                'timestamp': datetime.now()
            })
            
            # Process query
            with st.spinner("🔍 Searching documents..."):
                try:
                    result = self.rag.query(user_input.strip(), top_k=5)
                    
                    # Update last entry with answer
                    st.session_state.conversation_history[-1]['answer'] = result['answer']
                    st.session_state.conversation_history[-1]['sources'] = result.get('sources', [])
                    st.session_state.conversation_history[-1]['scores'] = result.get('scores', [])
                    
                except Exception as e:
                    st.session_state.conversation_history[-1]['answer'] = f"❌ Error: {str(e)}"
            
            st.rerun()


def create_streamlit_app(rag_system):
    """Create and run Streamlit app"""
    interface = StreamlitChatInterface(rag_system)
    interface.render()
