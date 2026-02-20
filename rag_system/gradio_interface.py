"""
Gradio Web Interface for RAG System
Provides a user-friendly web UI for querying documents
"""

import gradio as gr
from typing import Optional, List, Dict
from datetime import datetime


class GradioChatInterface:
    """Gradio-based web interface for the RAG system"""
    
    def __init__(self, rag_system):
        """
        Initialize the Gradio interface
        
        Args:
            rag_system: RAGSystem instance
        """
        self.rag = rag_system
        self.conversation_history = []
        self.interface = None
    
    def chat_response(self, message: str, history: List[List[str]]) -> tuple:
        """
        Process a chat message and return response
        
        Args:
            message: User message
            history: Chat history (list of [user_message, bot_response] pairs)
            
        Returns:
            Tuple of (empty string, updated history)
        """
        if not message or not message.strip():
            return "", history
        
        # Process query
        try:
            result = self.rag.query(message.strip(), top_k=5)
            
            answer = result['answer']
            sources = result.get('sources', [])
            scores = result.get('scores', [])
            
            # Format response with sources
            formatted_response = answer
            
            if sources:
                formatted_response += "\n\n**Sources:**\n"
                for i, source in enumerate(sources[:5], 1):  # Show top 5 sources
                    formatted_response += f"{i}. `{source}`\n"
            
            # Add to conversation history
            self.conversation_history.append({
                'query': message,
                'answer': answer,
                'sources': sources,
                'timestamp': datetime.now()
            })
            
            # Update history for Gradio
            history.append([message, formatted_response])
            
            return "", history
        
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            history.append([message, error_msg])
            return "", history
    
    def get_stats(self) -> str:
        """Get system statistics as formatted string"""
        stats = self.rag.get_stats()
        
        stats_text = f"""
## System Statistics

- **Total Chunks Indexed:** {stats['num_chunks']}
- **Embedding Dimension:** {stats['embedding_dim']}
- **Generator Type:** {stats['generator_type']}
- **Vector Store:** {stats.get('vector_store', 'ChromaDB')}
- **Framework:** {stats.get('framework', 'LangChain')}
- **Conversation Turns:** {len(self.conversation_history)}
"""
        return stats_text
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        return []
    
    def create_interface(self, share: bool = False, server_name: str = "127.0.0.1", server_port: int = 7860):
        """
        Create and launch the Gradio interface
        
        Args:
            share: Whether to create a public link
            server_name: Server hostname
            server_port: Server port
        """
        # Custom CSS for better styling
        css = """
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .chat-message {
            padding: 10px;
            margin: 5px 0;
        }
        """
        
        with gr.Blocks(title="RAG Chat Interface", theme=gr.themes.Soft(), css=css) as interface:
            gr.Markdown(
                """
                # 🤖 RAG Chat Interface
                Ask questions about your documents using Retrieval-Augmented Generation.
                
                **Features:**
                - 📚 Semantic search across your documents
                - 💬 Interactive chat interface
                - 📊 Source citations for answers
                - 🔍 Real-time document retrieval
                """
            )
            
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Chat",
                        height=500,
                        show_label=True,
                        avatar_images=(None, "🤖")
                    )
                    
            with gr.Row():
                msg = gr.Textbox(
                    label="Your Question",
                    placeholder="Type your question here...",
                    scale=4,
                    lines=2,
                    container=False
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
                    
                    with gr.Row():
                        clear_btn = gr.Button("Clear Chat", variant="secondary")
                        stats_btn = gr.Button("Show Statistics", variant="secondary")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 System Info")
                    stats_display = gr.Markdown(self.get_stats())
                    
                    gr.Markdown("### 💡 Tips")
                    gr.Markdown(
                        """
                        - Ask specific questions for better results
                        - Questions are answered based on your indexed documents
                        - Sources are shown below each answer
                        - Use "Clear Chat" to reset conversation
                        """
                    )
            
            # Event handlers
            msg.submit(
                self.chat_response,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )
            
            submit_btn.click(
                self.chat_response,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )
            
            clear_btn.click(
                self.clear_history,
                outputs=[chatbot]
            )
            
            stats_btn.click(
                lambda: self.get_stats(),
                outputs=[stats_display]
            )
        
        self.interface = interface
        return interface
    
    def launch(self, share: bool = False, server_name: str = "127.0.0.1", server_port: int = 7860, inbrowser: bool = True):
        """
        Launch the Gradio interface
        
        Args:
            share: Whether to create a public link
            server_name: Server hostname
            server_port: Server port
            inbrowser: Whether to open in browser automatically
        """
        if self.interface is None:
            self.create_interface(share=share, server_name=server_name, server_port=server_port)
        
        self.interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            inbrowser=inbrowser
        )


def create_gradio_interface(rag_system, **kwargs):
    """Factory function to create a Gradio interface"""
    interface = GradioChatInterface(rag_system)
    return interface
