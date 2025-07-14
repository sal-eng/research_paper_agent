#%% Import necessary libraries and loading environment variables

from pathlib import Path
from typing import Any
import tiktoken
import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from lancedb.table import LanceTable
from lancedb.rerankers import LinearCombinationReranker
import dotenv

dotenv.load_dotenv()





#%% LanceDB Embedding Model set up

model = get_registry().get("ollama").create(name="nomic-embed-text")



#%% Set up a DataModel Class to store documents in LanceDB

class Document(LanceModel):
    """
    Defines the schema for documents to be stored in LanceDB.
    """
    id: str
    text: str = model.SourceField()
    vector: Vector(model.ndims()) = model.VectorField()


#%% Chunk text function

def chunk_text(text: str, max_tokens: int = 1000, encoding_name: str = "cl100k_base") -> list[str]:
    """
    Splits the input text into chunks of a specified maximum token size.

    Args:
        text (str): The input text to be chunked.
        max_tokens (int, optional): The maximum number of tokens per chunk. Defaults to 1000.
        encoding_name (str, optional): The name of the encoding to use. Defaults to "cl100k_base".

    Returns:
        list[str]: A list of text chunks.
    """
    tokenizer = tiktoken.get_encoding(encoding_name)
    tokens = tokenizer.encode(text)

    for i in range(0, len(tokens), max_tokens):
        yield tokenizer.decode(tokens[i:i + max_tokens])

#%% Function to create a LanceDB table for storing documents

def create_documents_table(db_path: str, table_name: str, overwrite: bool = True) -> LanceTable:
    """
    Connect to LanceDB and create a table for storing documents.

    Args:
        db_path (str): The path to the LanceDB database.
        table_name (str): The name of the table to create.
        overwrite (bool): Whether to overwrite the table if it already exists.

    Returns:
        LanceTable: The created LanceDB table.
    """
    db = lancedb.connect(db_path)

    mode = "overwrite" if overwrite else "create"

    print(f"Creating table '{table_name}' in LanceDB at '{db_path}' with mode '{mode}'.")  

    table = db.create_table(
        table_name,
        schema=Document,
        mode=mode
    )

    table.create_fts_index("text", replace=overwrite)

    return table

#%% Function to delete a LanceDB table

def delete_documents_table(db_path: str, table_name: str) -> None:
    """
    Deletes a LanceDB table for storing documents.

    Args:
        db_path (str): The path to the LanceDB database.
        table_name (str): The name of the table to delete.
    """
    db = lancedb.connect(db_path)
    db.drop_table(table_name, ignore_missing=True)

# %% Function to add markdown documents from a local directory to a LanceDB table

def add_docs_to_table(
    table: LanceTable,
    docs_dir: str,
    max_tokens: int = 8192
) -> None:
    """
    Adds markdown documents from a local directory to a LanceDB table.

    Args:
        table (LanceTable): The LanceDB table to which documents will be added.
        docs_dir (str): The directory containing markdown files to be added.
        max_tokens (int, optional): The maximum number of tokens per chunk. Defaults to 8192.

    Returns:
        none
    """
    docs = []
    knowledge_base = Path(docs_dir)

    for md_file in knowledge_base.glob("*.md"):
        print(f"Processing file: {md_file}")
        with open(md_file, "r", encoding="utf-8") as f:
            text = f.read()
            for i, chunk in enumerate(chunk_text(text, max_tokens=max_tokens)):
                doc_id = f"{md_file.stem}_{i}"  # Unique ID for each chunk
                docs.append({
                    "id": doc_id,
                    "text": chunk
                })
    
    if docs:
        # Create vectors for the documents
        table.add(docs)
        print(f"Adding {len(docs)} documents to the table.")
    else:
        print("No documents to add.")

# %% Function to retrieve documents from a LanceDB table based on a query

def retrieve_similar_docs(
    table: LanceTable,
    query: str,
    query_type: str = "hybrid",
    limit: int = 100,
    reranker_weight: float = 0.7,
) -> list[dict[str, Any]]:
    """
    Retrieves similar documents from a LanceDB table based on a query.
    Uses a hybrid search approach with reranking.

    Args:
        table (LanceTable): The LanceDB table from which to retrieve documents.
        query (str): The query string to search for.
        query_type (str, optional): The type of query to perform. Defaults to "hybrid".
        limit (int, optional): The maximum number of documents to retrieve. Defaults to 100.
        reranker_weight (float, optional): The weight for the reranker. Defaults to 0.7.

    Returns:
        list[dict[str, Any]]: A list of dictionaries containing the retrieved documents.
    """

    reranker = LinearCombinationReranker(weight=reranker_weight)

    results = (
        table.search(query, query_type=query_type)
        .rerank(reranker=reranker)
        .limit(limit)
        .to_list()
    )

    return results

# %% Function to set up a LanceDB database with a documents table
def setup_lancedb() -> None:
    """
    Sets up a LanceDB database with a documents table.

    Args:
        None

    Returns:
        None
    """
    db_path = "./lance_db"
    table_name = "knowledge"
    knowledge_base_dir = Path("./knowledge_markdowns")

    # Create the lancedb table
    table = create_documents_table(db_path, table_name, overwrite=True)

    # Add documents to the table from the knowledge base directory
    add_docs_to_table(table, knowledge_base_dir)

    

# %%
# Run the setup function to create the LanceDB database and table
if __name__ == "__main__":
    setup_lancedb()
    print("LanceDB setup complete.") 

    # test the retrieval function
    db = lancedb.connect("./lance_db")
    table = db.open_table("knowledge")
    query = "What is the attention mechanism?"
    results = retrieve_similar_docs(table, query, limit=5)
    print(f"Retrieved {len(results)} documents for query '{query}':")
    for doc in results:
        print(f"- {doc['text'][:100]}...")
    # This will print the first 100 characters of each retrieved document
    # to verify the setup and retrieval functionality.
    # You can adjust the query and limit as needed to test different scenarios.
    # This is a simple test to ensure that the setup and retrieval functions work as expected.
    # You can expand this with more complex queries or additional functionality as needed.
    # Note: Ensure that the knowledge_markdowns directory exists and contains markdown files for testing.
    

# %%
