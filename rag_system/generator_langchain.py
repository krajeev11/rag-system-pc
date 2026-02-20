"""
Generator Module using LangChain
Generates responses using LangChain LLMs with retrieved context
"""

import os
from typing import List, Dict, Optional
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


class LangChainGenerator:
    """Generates text responses using LangChain LLMs with retrieved context"""
    
    def __init__(
        self, 
        model_type: str = "simple",
        api_key: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo",
        vectorstore=None
    ):
        """
        Initialize the generator
        
        Args:
            model_type: Type of model to use ("simple", "openai", "anthropic", "local")
            api_key: API key for the model (if required)
            model_name: Name of the model to use
            vectorstore: LangChain vectorstore for RAG chain (optional)
        """
        self.model_type = model_type
        self.api_key = api_key
        self.model_name = model_name
        self.vectorstore = vectorstore
        self.llm = None
        self.qa_chain = None
        
        if model_type != "simple":
            self._init_llm()
    
    def _init_llm(self):
        """Initialize the LangChain LLM"""
        try:
            if self.model_type == "openai":
                from langchain_openai import ChatOpenAI
                api_key = self.api_key or os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key is required")
                self.llm = ChatOpenAI(
                    model_name=self.model_name,
                    openai_api_key=api_key,
                    temperature=0.7
                )
            elif self.model_type == "anthropic":
                from langchain_anthropic import ChatAnthropic
                api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("Anthropic API key is required")
                self.llm = ChatAnthropic(
                    model=self.model_name,
                    anthropic_api_key=api_key,
                    temperature=0.7
                )
            elif self.model_type == "local":
                # For local models using HuggingFace
                from langchain_community.llms import HuggingFacePipeline
                # This is a placeholder - would need proper setup
                self.llm = None
        except ImportError as e:
            raise ImportError(
                f"Required package not installed. For OpenAI: pip install langchain-openai. "
                f"For Anthropic: pip install langchain-anthropic. Error: {e}"
            )
    
    def generate(
        self,
        query: str,
        context_chunks: List[Dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> str:
        """
        Generate a response based on query and context
        
        Args:
            query: User query
            context_chunks: List of retrieved context chunks
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response text
        """
        if self.model_type == "simple" or self.llm is None:
            return self._simple_generate(query, context_chunks)
        
        # Build context from chunks
        context = self._build_context(context_chunks)
        
        # Build prompt
        prompt = self._build_prompt(query, context)
        
        # Generate using LLM
        try:
            response = self.llm.invoke(prompt)
            if hasattr(response, 'content'):
                return response.content
            return str(response)
        except Exception as e:
            return f"I would answer: {prompt[:200]}... (Note: LLM call failed: {str(e)})"
    
    def _build_context(self, chunks: List[Dict[str, str]]) -> str:
        """Build context string from chunks"""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get('source', 'Unknown')
            content = chunk.get('content', '')
            context_parts.append(f"[Document {i} from {source}]\n{content}\n")
        return "\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build the prompt for the LLM"""
        return f"""You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {query}

Answer the question based on the context above. If the context doesn't contain enough information to answer the question, say so."""
    
    def _simple_generate(self, query: str, chunks: List[Dict[str, str]]) -> str:
        """Simple template-based answer generation"""
        if not chunks:
            return f"I don't have enough information to answer: {query}"
        
        # Use the top chunk to generate a simple answer
        top_chunk = chunks[0]
        source = top_chunk.get('source', 'Unknown')
        content = top_chunk.get('content', '')
        
        answer = f"Based on the information from {source}:\n\n{content}"
        
        # Add additional context if available
        if len(chunks) > 1:
            answer += f"\n\nAdditional relevant information:\n"
            for i, chunk in enumerate(chunks[1:3], 2):  # Include up to 2 more chunks
                answer += f"\n[{i}] {chunk.get('content', '')[:200]}...\n"
        
        return answer
    
    def create_qa_chain(self, vectorstore, prompt_template: Optional[str] = None):
        """
        Create a RetrievalQA chain for better RAG integration
        
        Args:
            vectorstore: LangChain vectorstore
            prompt_template: Optional custom prompt template
        """
        if self.llm is None:
            raise ValueError("LLM must be initialized before creating QA chain")
        
        if prompt_template is None:
            prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        return self.qa_chain
    
    def generate_with_chain(self, query: str) -> Dict:
        """
        Generate response using QA chain (requires create_qa_chain to be called first)
        
        Args:
            query: User query
            
        Returns:
            Dictionary with 'answer' and 'sources'
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not created. Call create_qa_chain first.")
        
        result = self.qa_chain.invoke({"query": query})
        
        return {
            'answer': result['result'],
            'sources': [doc.metadata.get('source', 'Unknown') for doc in result.get('source_documents', [])]
        }
