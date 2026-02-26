#!/usr/bin/env python3
"""
Sample script demonstrating how to use the Together API-based ChatPDF implementation with local embeddings.
No OpenAI API key required!
"""
import os
import argparse
from methods.together_local.together_method_local import TogetherChatPDFLocal

def main():
    parser = argparse.ArgumentParser(description="Chat with a PDF using Together API-based RAG with local embeddings")
    parser.add_argument("--pdf", type=str, required=True, help="Path to the PDF file")
    parser.add_argument("--together_api_key", type=str, 
                        default="fa80595b027f7af95287cb1ec86b2650fd7f09a63e71a596390860cafa191ed5", 
                        help="Together API key (or set TOGETHER_API_KEY env var)")
    parser.add_argument("--model", type=str, 
                        default="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free", 
                        help="Together model name")
    parser.add_argument("--embedding_model", type=str, 
                        default="all-MiniLM-L6-v2", 
                        help="HuggingFace embedding model")
    parser.add_argument("--k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--threshold", type=float, default=0.2, help="Similarity threshold")
    
    args = parser.parse_args()
    
    # Set the environment variable for the Together API key
    os.environ["TOGETHER_API_KEY"] = args.together_api_key
    
    # Initialize the Together API-based ChatPDF with local embeddings
    print("Initializing TogetherChatPDFLocal with local embeddings...")
    chat_pdf = TogetherChatPDFLocal(
        together_api_key=args.together_api_key,
        model=args.model,
        embedding_model=args.embedding_model
    )
    
    # Ingest the PDF
    print(f"Ingesting PDF: {args.pdf}")
    chat_pdf.ingest(args.pdf)
    print("Ingestion complete!")
    
    # Interactive chat loop
    print("\nChat with your PDF using Together AI (type 'exit' to quit):")
    while True:
        query = input("\nYour question: ")
        if query.lower() in ["exit", "quit", "q"]:
            break
        
        # Get the answer
        print("Getting answer from Together AI...")
        answer = chat_pdf.ask(
            query=query,
            k=args.k,
            score_threshold=args.threshold
        )
        
        print(f"\nAnswer: {answer}")
    
    print("\nThank you for using Together API-based ChatPDF with local embeddings!")

if __name__ == "__main__":
    main() 