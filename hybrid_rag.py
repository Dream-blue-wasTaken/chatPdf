# hybrid_rag.py
from langchain_core.globals import set_verbose, set_debug
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
from together import Together
import logging
import os

set_debug(True)
set_verbose(True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridChatPDF:
    """
    A class for handling PDF ingestion and question answering using RAG.
    Uses local embeddings but Together API for LLM responses.
    """

    def __init__(
        self, 
        together_api_key: str = "fa80595b027f7af95287cb1ec86b2650fd7f09a63e71a596390860cafa191ed5",
        together_model: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the HybridChatPDF instance with Together API and local embeddings.
        
        Args:
            together_api_key: Together API key
            together_model: Model ID on Together (default: DeepSeek R1)
            embedding_model: HuggingFace embedding model name
        """
        # Initialize Together API client
        self.together_api_key = together_api_key or os.environ.get("TOGETHER_API_KEY")
        self.together_model = together_model
        self.together_client = Together(api_key=self.together_api_key)
        
        # Initialize local embeddings using HuggingFaceEmbeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        
        # Prompt template (same as original)
        self.prompt_template = """
        You are a helpful assistant answering questions based on the uploaded document.
        Do not show the thinking process.
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
        """
        logger.info(f"Starting ingestion for file: {pdf_file_path}")
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="chroma_db_hybrid",
        )
        logger.info("Ingestion completed. Document embeddings stored successfully.")

    def ask(self, query: str, k: int = 5, score_threshold: float = 0.2):
        """
        Answer a query using the RAG pipeline with local retrieval but Together API for generation.
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

        # Format the context from retrieved documents
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        
        # Format the prompt with retrieved context and query
        formatted_prompt = self.prompt_template.format(context=context, question=query)
        
        # Call Together API
        logger.info("Generating response using Together API.")
        response = self.together_client.chat.completions.create(
            model=self.together_model,
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