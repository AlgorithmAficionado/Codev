# llm_processing.py
import os
from groq import Groq
from .input_layer import prepare_query

def generate_response_with_groq(query, context=None, model=None):
    """
    Generate a response using the specified Groq model.

    Args:
        query (str): The user's input query.
        context (str, optional): Relevant context for the query.
        model (str, optional): The Groq model to use.

    Returns:
        str: The generated response from Groq.
    """
    try:
        model = model or os.getenv('GROQ_MODEL')
        api_key = os.getenv('GROQ_API_KEY')

        if not api_key:
            raise ValueError("API key is missing. Please set the GROQ_API_KEY environment variable.")

        client = Groq(api_key=api_key)
        prepared_query = prepare_query(query, context)

        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prepared_query}],
            model=model,
        )

        return chat_completion.choices[0].message.content

    except ValueError as ve:
        print(f"ValueError: {ve}")
        return "There was an issue with your request."

    except Exception as e:
        print(f"An error occurred: {e}")
        return "An error occurred while processing your request."
