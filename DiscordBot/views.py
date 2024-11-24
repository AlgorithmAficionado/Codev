from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .serializers import ChatSerializer
from .models import Chats
from Rag.Utility.knowledge_base import get_relevant_context
from .utility import create_response
from Rag.Utility.llm_processing_layer import generate_response_with_groq


class ChatsViewSet(viewsets.ModelViewSet):
    serializer_class = ChatSerializer
    queryset = Chats.objects.all()

    @action(detail=False, methods=['POST'])
    def generate_response(self, request):
        try:
            # Extract the query from the request
            query = request.data.get('query')
            if not query:
                return create_response(
                    success=False,
                    message="Query is required.",
                    body=None,
                    status_code=status.HTTP_400_BAD_REQUEST
                )

            # Fetch chat history from the database
            chat_history = Chats.objects.all()

            # Convert QuerySet to list of valid documents
            documents = []
            for record in chat_history:
                # Ensure all required fields are non-empty strings
                if (
                    isinstance(record.query, str) and record.query.strip() and
                    isinstance(record.response, str) and record.response.strip() and
                    isinstance(record.context, str) and record.context.strip()
                ):
                    documents.append(
                        f"Query: {record.query}, Response: {record.response}, Context: {record.context}"
                    )
            print(f"Filtered Documents: {documents}")

            # Provide default history if no valid records are found
            if not documents:
                documents = ["This is the first interaction. No history available."]
            
            # Get relevant context from the knowledge base
            context = get_relevant_context(documents, query)
            print(f"Relevant Context: {context}")
            
            # Generate context and response
            response = generate_response_with_groq(query, context)

            # Save the new chat in the database
            Chats.objects.create(query=query, response=response, context=context)

            # Return the response
            return create_response(
                success=True,
                message="Successfully generated response",
                body=response,
                status_code=status.HTTP_200_OK
            )

        except Exception as e:
            # Handle unexpected errors
            return create_response(
                success=False,
                message=f"An error occurred: {str(e)}",
                body=None,
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
