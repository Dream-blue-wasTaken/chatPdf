#!/usr/bin/env python3
"""
Sample script demonstrating how to use the Together API-based ChatPDF implementation.
"""
import os
import argparse
from together_method import TogetherChatPDF

def main():
    parser = argparse.ArgumentParser(description="Chat with a PDF using Together API-based RAG")
    parser.add_argument("--pdf", type=str, required=True, help="Path to the PDF file")
    parser.add_argument("--together_api_key", type=str, 
                        default="fa80595b027f7af95287cb1ec86b2650fd7f09a63e71a596390860cafa191ed5", 
                        help="Together API key (or set TOGETHER_API_KEY env var)")
    parser.add_argument("--openai_api_key", type=str, help="OpenAI API key for embeddings (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", type=str, 
                        default="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free", 
                        help="Together model name")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-ada-002", 
                        help="OpenAI embedding model")
    parser.add_argument("--k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--threshold", type=float, default=0.2, help="Similarity threshold")
    
    args = parser.parse_args()
    
    # Set the environment variable for the Together API key
    os.environ["TOGETHER_API_KEY"] = args.together_api_key
    
    # Initialize the Together API-based ChatPDF
    chat_pdf = TogetherChatPDF(
        together_api_key=args.together_api_key,
        model=args.model,
        embedding_api_key=args.openai_api_key,
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
        answer = chat_pdf.ask(
            query=query,
            k=args.k,
            score_threshold=args.threshold
        )
        
        print(f"\nAnswer: {answer}")
    
    print("\nThank you for using Together API-based ChatPDF!")

if __name__ == "__main__":
    main() 