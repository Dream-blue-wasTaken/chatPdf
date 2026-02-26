"""
Implementation of ChatPDF using the Together API with DeepSeek R1 Distill Llama 70B model.
This module provides similar functionality to alternative_method.py but uses Together's API.
"""
import os
import logging
from typing import List, Dict, Any, Optional

from together import Together
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_openai import OpenAIEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TogetherChatPDF:
    """A class for handling PDF ingestion and question answering using RAG with Together API."""

    def __init__(
        self,
        together_api_key: Optional[str] = None,
        model: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
        embedding_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-ada-002"
    ):
        """
        Initialize the TogetherChatPDF instance with API credentials.
        
        Args:
            together_api_key: Together API key. If None, will look for TOGETHER_API_KEY in environment.
            model: Together model ID to use
            embedding_api_key: OpenAI API key for embeddings. If None, will look for OPENAI_API_KEY in environment.
            embedding_model: OpenAI model to use for embeddings
        """
        self.together_api_key = together_api_key or os.environ.get("TOGETHER_API_KEY")
        self.embedding_api_key = embedding_api_key or os.environ.get("OPENAI_API_KEY")
        self.model_name = model
        
        if not self.together_api_key:
            raise ValueError(
                "Together API key is required. Either pass it as together_api_key or "
                "set the TOGETHER_API_KEY environment variable."
            )
        
        if not self.embedding_api_key:
            raise ValueError(
                "OpenAI API key for embeddings is required. Either pass it as embedding_api_key or "
                "set the OPENAI_API_KEY environment variable."
            )
        
        # Initialize Together client
        self.together_client = Together(api_key=self.together_api_key)
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=self.embedding_api_key
        )
        
        # Text splitter for document chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, 
            chunk_overlap=100
        )
        
        # Prompt template for RAG
        self.prompt_template = """
        You are a helpful assistant answering questions based on the uploaded document.
        Context:
        {context}
        
        Question:
        {question}
        
        Answer concisely and accurately in three sentences or less.
        """
        
        self.vector_store = None
        self.retriever = None

    def ingest(self, pdf_file_path: str):
        """
        Ingest a PDF file, split its contents, and store the embeddings in the vector store.
        
        Args:
            pdf_file_path: Path to the PDF file to ingest
        """
        logger.info(f"Starting ingestion for file: {pdf_file_path}")
        
        # Load and split the document
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="chroma_db_together",  # Use a different directory
        )
        logger.info("Ingestion completed. Document embeddings stored successfully.")

    def ask(self, query: str, k: int = 5, score_threshold: float = 0.2):
        """
        Answer a query using the RAG pipeline with Together API.
        
        Args:
            query: The user's question
            k: Number of documents to retrieve
            score_threshold: Similarity threshold for retrieval
            
        Returns:
            str: The answer to the query
        """
        if not self.vector_store:
            raise ValueError("No vector store found. Please ingest a document first.")

        if not self.retriever:
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": k, "score_threshold": score_threshold},
            )

        logger.info(f"Retrieving context for query: {query}")
        retrieved_docs = self.retriever.invoke(query)

        if not retrieved_docs:
            return "No relevant context found in the document to answer your question."

        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        
        # Format the prompt with retrieved context and query
        formatted_prompt = self.prompt_template.format(context=context, question=query)
        
        # Call Together API
        logger.info("Generating response using the Together API.")
        response = self.together_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": formatted_prompt}
            ]
        )
        
        return response.choices[0].message.content

    def clear(self):
        """
        Reset the vector store and retriever.
        """
        logger.info("Clearing vector store and retriever.")
        self.vector_store = None
        self.retriever = None 