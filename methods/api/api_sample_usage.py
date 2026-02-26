#!/usr/bin/env python3
"""
Sample script demonstrating how to use the API-based ChatPDF implementation.
"""
import os
import argparse
from methods.api.alternative_method import APIChatPDF

def main():
    parser = argparse.ArgumentParser(description="Chat with a PDF using API-based RAG")
    parser.add_argument("--pdf", type=str, required=True, help="Path to the PDF file")
    parser.add_argument("--openai_api_key", type=str, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="OpenAI model name")
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-small-en-v1.5", help="Hugging Face embedding model")
    parser.add_argument("--k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--threshold", type=float, default=0.2, help="Similarity threshold")
    
    args = parser.parse_args()
    
    # Initialize the API-based ChatPDF
    chat_pdf = APIChatPDF(
        openai_api_key=args.openai_api_key,
        openai_model=args.model,
        embedding_model=args.embedding_model
    )
    
    # Ingest the PDF
    print(f"Ingesting PDF: {args.pdf}")
    chat_pdf.ingest(args.pdf)
    print("Ingestion complete!")
    
    # Interactive chat loop
    print("\nChat with your PDF (type 'exit' to quit):")
    while True:
        query = input("\nYour question: ")
        if query.lower() in ["exit", "quit", "q"]:
            break
        
        # Get the answer
        answer = chat_pdf.ask(
            query=query,
            k=args.k,
            score_threshold=args.threshold
        )
        
        print(f"\nAnswer: {answer}")
    
    print("\nThank you for using API-based ChatPDF!")

if __name__ == "__main__":
    main() 