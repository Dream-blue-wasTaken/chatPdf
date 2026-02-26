#!/usr/bin/env python3
"""
Simple example of using the Together API with DeepSeek R1 Distill Llama 70B model.
"""
from together import Together

# Initialize the Together API client with the API key
API_KEY = "fa80595b027f7af95287cb1ec86b2650fd7f09a63e71a596390860cafa191ed5"
client = Together(api_key=API_KEY)

# Create a completion using the DeepSeek R1 Distill Llama 70B model
response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
    messages=[
        {
            "role": "user",
            "content": "What are some fun things to do in New York?"
        }
    ]
)

# Print the response
print(response.choices[0].message.content)

# Example of providing context from a PDF for a RAG-like setup
def ask_with_context(question, context):
    """Ask a question with context from a document."""
    prompt = f"""
    Use the following context to answer the question:
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    return response.choices[0].message.content

# Example usage with some sample context
if __name__ == "__main__":
    sample_context = """
    New York City is composed of five boroughs: The Bronx, Brooklyn, Manhattan, Queens, and Staten Island.
    The Empire State Building is a 102-story skyscraper located in Midtown Manhattan.
    Central Park is an urban park in Manhattan, New York City. It is located between the Upper West Side 
    and Upper East Side, roughly bounded by Fifth Avenue on the east, Central Park West on the west, 
    Central Park South on the south, and Central Park North on the north.
    The Metropolitan Museum of Art, colloquially "the Met", is the largest art museum in the United States.
    """
    
    question = "What are some famous landmarks in New York City?"
    
    print("\nQuestion with context:")
    print(f"Q: {question}")
    print(f"A: {ask_with_context(question, sample_context)}") 