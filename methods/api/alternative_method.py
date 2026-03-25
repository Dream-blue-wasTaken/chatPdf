"""
Alternative implementation of ChatPDF using API services instead of local models.
This module provides the same interface as the original rag.py but uses remote APIs.
"""
import os
import logging
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIChatPDF:
    """A class for handling PDF ingestion and question answering using RAG with API services."""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-3.5-turbo",
        openai_api_base: Optional[str] = None,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        persist_directory: str = "chroma_db_api",
    ):
        """
        Initialize the APIChatPDF instance with API credentials.
        
        Args:
            openai_api_key: OpenAI API key. If None, will look for OPENAI_API_KEY in environment.
            openai_model: OpenAI model to use for chat completions
            openai_api_base: Base URL for OpenAI-compatible API
            embedding_model: Hugging Face model to use for local embeddings
            persist_directory: Path for persisted Chroma database files
        """
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key is required. Either pass it as openai_api_key or "
                "set the OPENAI_API_KEY environment variable."
            )
        
        # Initialize language model and embeddings
        chat_kwargs = {
            "model": openai_model,
            "api_key": self.openai_api_key,
            "temperature": 0
        }
        if openai_api_base:
            chat_kwargs["base_url"] = openai_api_base
            
        self.model = ChatOpenAI(**chat_kwargs)
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model
        )
        
        # Text splitter for document chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, 
            chunk_overlap=100
        )
        
        # Prompt template for RAG
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful assistant answering questions based on the uploaded document.
            Context:
            {context}
            
            Question:
            {question}
            
            Answer concisely and accurately in three sentences or less.
            """
        )
        self.persist_directory = persist_directory
        self.vector_store = None

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

        if not chunks:
            raise ValueError("No text could be extracted from the PDF.")

        # Create or append to the vector store
        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
            )
        else:
            self.vector_store.add_documents(chunks)
        logger.info("Ingestion completed. Document embeddings stored successfully.")

    def ask(self, query: str, k: int = 5, score_threshold: float = 0.2):
        """
        Answer a query using the RAG pipeline with API models.
        
        Args:
            query: The user's question
            k: Number of documents to retrieve
            score_threshold: Similarity threshold for retrieval
            
        Returns:
            str: The answer to the query
        """
        if not self.vector_store:
            raise ValueError("No vector store found. Please ingest a document first.")

        retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k, "score_threshold": score_threshold},
        )

        logger.info(f"Retrieving context for query: {query}")
        retrieved_docs = retriever.invoke(query)

        if not retrieved_docs:
            return "No relevant context found in the document to answer your question."

        formatted_input = {
            "context": "\n\n".join(doc.page_content for doc in retrieved_docs),
            "question": query,
        }

        # Build the RAG chain
        chain = (
            RunnablePassthrough()  # Passes the input as-is
            | self.prompt           # Formats the input for the LLM
            | self.model            # Queries the LLM
            | StrOutputParser()     # Parses the LLM's output
        )

        logger.info("Generating response using the API.")
        return chain.invoke(formatted_input)

    def clear(self):
        """
        Reset the vector store and retriever.
        """
        logger.info("Clearing vector store and retriever.")
        if self.vector_store is not None:
            self.vector_store.delete_collection()
        self.vector_store = None
