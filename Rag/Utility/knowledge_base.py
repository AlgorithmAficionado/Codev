from astrapy import DataAPIClient


def initialize_astra_client(api_endpoint, token):
    """
    Initialize Astra DB client.

    Args:
        api_endpoint (str): The Astra DB API endpoint.
        token (str): The token for authentication.

    Returns:
        DataAPIClient: The initialized Astra DB client.
    """
    try:
        client = DataAPIClient(token)
        db = client.get_database_by_api_endpoint(api_endpoint)
        print(f"Connected to Astra DB.")
        return db
    except Exception as e:
        print(f"Error initializing Astra client: {e}")
        return None


def check_and_create_table(db, keyspace, table_name, vector_dimension=768):
    """
    Check if a collection (table) exists in Astra DB. If it doesn't exist, create it.

    Args:
        db (DataAPIClient): Initialized Astra DB client.
        keyspace (str): The keyspace where the table resides or will be created.
        table_name (str): The name of the table to check/create.
        vector_dimension (int): Dimension of the vector column (default: 768).

    Returns:
        str: Success message or error details.
    """
    try:
        # Check if the table exists
        check_query = f"""
        SELECT table_name
        FROM system_schema.tables
        WHERE keyspace_name = '{keyspace}' AND table_name = '{table_name}';
        """
        existing_tables = db.cql(check_query)

        if existing_tables:
            print(f"Table '{table_name}' already exists in keyspace '{keyspace}'.")
            return f"Table '{table_name}' already exists in keyspace '{keyspace}'."

        # Create the table if it does not exist
        create_table_query = f"""
        CREATE TABLE {keyspace}.{table_name} (
            id UUID PRIMARY KEY,
            document TEXT,
            metadata MAP<TEXT, TEXT>,
            embedding VECTOR<FLOAT, {vector_dimension}>
        );
        """
        db.cql(create_table_query)
        print(f"Table '{table_name}' successfully created in keyspace '{keyspace}'.")
        return f"Table '{table_name}' successfully created in keyspace '{keyspace}'."

    except Exception as e:
        return f"Error checking or creating table: {e}"


def add_or_update_documents(db, keyspace, table_name, documents, metadata_list=None):
    """
    Add or update documents in Astra DB.

    Args:
        db (DataAPIClient): Initialized Astra DB client.
        keyspace (str): The keyspace where the table resides.
        table_name (str): The name of the table.
        documents (list): List of documents to store.
        metadata_list (list of dict, optional): Metadata for each document.

    Returns:
        str: Success message or error details.
    """
    if metadata_list is None:
        metadata_list = [{}] * len(documents)

    try:
        # Insert or update each document
        for doc, metadata in zip(documents, metadata_list):
            # Replace or insert the document into the table
            upsert_query = f"""
            INSERT INTO {keyspace}.{table_name} (id, document, metadata, embedding)
            VALUES (now(), %s, %s, %s);
            """
            embedding = [0.0] * 768  # Placeholder for embeddings
            db.cql(upsert_query, [doc, metadata, embedding])
            print(f"Document added or updated: {doc[:50]}...")
        return "Documents added or updated successfully."
    except Exception as e:
        return f"Error adding or updating documents: {e}"


def search_documents(db, keyspace, table_name, user_query, top_k=3):
    """
    Perform a vector search for context.

    Args:
        db (DataAPIClient): Initialized Astra DB client.
        keyspace (str): The keyspace where the table resides.
        table_name (str): The name of the table.
        user_query (str): The query from the user to search for relevant context.
        top_k (int): Number of top results to retrieve.

    Returns:
        list: Relevant context documents retrieved based on the query.
    """
    try:
        # Perform vector search
        search_query = f"""
        SELECT * FROM {keyspace}.{table_name}
        WHERE embedding ANN OF %s LIMIT {top_k};
        """
        search_results = db.cql(search_query, [user_query])

        # Collect relevant context from search results
        relevant_context = []
        for result in search_results:
            relevant_context.append(result["document"])

        return relevant_context
    except Exception as e:
        print(f"Error during vector search: {e}")
        return []
