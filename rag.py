import logging
from typing import Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

logger = logging.getLogger(__name__)


class ChatPDF:
    """Handles PDF ingestion and question answering with a RAG pipeline."""

    def __init__(
        self,
        llm_model: str = "deepseek-r1:latest",
        embedding_model: str = "mxbai-embed-large",
        persist_directory: str = "chroma_db",
    ):
        self.model = ChatOllama(model=llm_model)
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.persist_directory = persist_directory
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful assistant answering questions based on the uploaded document.
            Do not show the thinking process.

            Context:
            {context}

            Question:
            {question}

            Answer concisely and accurately in three sentences or less.
            """
        )

        self.vector_store: Optional[Chroma] = None

    def ingest(self, pdf_file_path: str) -> None:
        """Ingest a PDF file, split it into chunks, and store embeddings in Chroma."""
        logger.info("Starting ingestion for file: %s", pdf_file_path)
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )
        logger.info("Ingestion completed and embeddings stored successfully.")

    def ask(self, query: str, k: int = 5, score_threshold: float = 0.2) -> str:
        """Answer a query using retrieved document context and the configured model."""
        if not self.vector_store:
            raise ValueError("No vector store found. Please ingest a document first.")

        retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k, "score_threshold": score_threshold},
        )

        logger.info("Retrieving context for query: %s", query)
        retrieved_docs = retriever.invoke(query)

        if not retrieved_docs:
            return "No relevant context found in the document to answer your question."

        formatted_input = {
            "context": "\n\n".join(doc.page_content for doc in retrieved_docs),
            "question": query,
        }

        chain = (
            RunnablePassthrough()
            | self.prompt
            | self.model
            | StrOutputParser()
        )

        logger.info("Generating response using the LLM.")
        return chain.invoke(formatted_input)

    def clear(self) -> None:
        """Reset in-memory vector store reference."""
        logger.info("Clearing vector store reference.")
        self.vector_store = None
