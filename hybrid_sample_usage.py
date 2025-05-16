#!/usr/bin/env python3
"""
Sample script demonstrating the hybrid approach:
- Local embeddings using HuggingFace sentence-transformers
- Together API for LLM responses
"""
import os
import argparse
from hybrid_rag import HybridChatPDF

def main():
    parser = argparse.ArgumentParser(description="Chat with a PDF using hybrid RAG (HuggingFace embeddings + Together API)")
    parser.add_argument("--pdf", type=str, required=True, help="Path to the PDF file")
    parser.add_argument("--together_api_key", type=str, 
                        default="fa80595b027f7af95287cb1ec86b2650fd7f09a63e71a596390860cafa191ed5", 
                        help="Together API key (or set TOGETHER_API_KEY env var)")
    parser.add_argument("--together_model", type=str, 
                        default="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free", 
                        help="Together model name")
    parser.add_argument("--embedding_model", type=str, 
                        default="all-MiniLM-L6-v2", 
                        help="HuggingFace embedding model name")
    parser.add_argument("--k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--threshold", type=float, default=0.2, help="Similarity threshold")
    
    args = parser.parse_args()
    
    # Set the environment variable for the Together API key
    os.environ["TOGETHER_API_KEY"] = args.together_api_key
    
    # Initialize the hybrid ChatPDF system
    print("Initializing HybridChatPDF with HuggingFace embeddings and Together API for responses...")
    print(f"Using embedding model: {args.embedding_model}")
    print(f"Using LLM model: {args.together_model}")
    
    chat_pdf = HybridChatPDF(
        together_api_key=args.together_api_key,
        together_model=args.together_model,
        embedding_model=args.embedding_model
    )
    
    # Ingest the PDF
    print(f"Ingesting PDF: {args.pdf}")
    chat_pdf.ingest(args.pdf)
    print("Ingestion complete!")
    
    # Interactive chat loop
    print("\nChat with your PDF using Together API (type 'exit' to quit):")
    while True:
        query = input("\nYour question: ")
        if query.lower() in ["exit", "quit", "q"]:
            break
        
        # Get the answer
        print("Getting answer...")
        answer = chat_pdf.ask(
            query=query,
            k=args.k,
            score_threshold=args.threshold
        )
        
        print(f"\nAnswer: {answer}")
    
    print("\nThank you for using HybridChatPDF!")

if __name__ == "__main__":
    main() 