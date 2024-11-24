# input_layer.py
from langchain.prompts import PromptTemplate

# Define a LangChain PromptTemplate for formatting the query and context
prompt_template = PromptTemplate(
    input_variables=["context", "query"],
    template="Context: {context}\nQuery: {query}\nResponse:"
)

def prepare_query(query, context=None):
    """
    Prepares the user query by adding context and formatting it into a template.

    Args:
        query (str): The user's input query.
        context (str, optional): Relevant context to augment the query.

    Returns:
        str: The formatted query with context, ready for LLM input.
    """
    # Format the query and context into a structured prompt
    if context:
        formatted_prompt = prompt_template.format(context=context, query=query)
    else:
        formatted_prompt = f"Query: {query}\nResponse:"
    
    return formatted_prompt
