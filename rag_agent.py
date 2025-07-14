#%% Import necessary libraries and loading environment variables

import lancedb
from pydantic_ai import Agent, RunContext, ModelRetry
from httpx import AsyncClient
from load_models import GEMINI_MODEL
from lancedb_setup import setup_lancedb, retrieve_similar_docs
import dotenv
import asyncio

dotenv.load_dotenv()


#%% Set up the Knowledge Query Agent that generates a query string from the user input

def setup_knowledge_query_agent() -> Agent:
    """
    Sets up the Knowledge Query Agent that generates a query string from the user input.
    
    Returns:
        Agent: The configured Knowledge Query Agent.
    """
    knowledge_query_agent = Agent(
        name="Knowledge Query Agent",
        model=GEMINI_MODEL,
        instructions="""
        From the input string, generate a query string to pass to the knowledge base.
        """,
        # deps_type=Deps,
        # result_type=str,
    )    

    return knowledge_query_agent

  


#%% Set up the main agent

def setup_rag_agent() -> Agent:
    """
    Sets up the RAG Agent that retrieves similar documents from the knowledge base.
    
    Returns:
        Agent: The configured RAG Agent.
    """
    rag_agent = Agent(
        name="RAG Agent",
        model=GEMINI_MODEL,
        instructions="""
        You're a helpful assistant. 
        """
    )

    return rag_agent
# %%



async def main():
    async with AsyncClient() as client:
        """
        Main function to set up the RAG Agent and run it.
        """
        db_path = "./lance_db"
        table_name = "knowledge"
        db = lancedb.connect(db_path)

        knowledge_table = db.open_table(table_name) if db.table_names() else setup_lancedb()

        knowledge_query_agent = setup_knowledge_query_agent()

        rag_agent = setup_rag_agent()

        message_history = None

        while True:
            query = input("Enter your query (or 'exit' to quit): ")
            if query.lower() == 'exit':
                break

            res = await knowledge_query_agent.run(query)
            knowledge_query = res.output
            print(f"Generated Knowledge Query: {knowledge_query}")

            similar_docs = retrieve_similar_docs( 
                knowledge_table,
                query=knowledge_query,
                query_type="hybrid",
                limit=5,
                reranker_weight=0.7
            )

            if not similar_docs:
                print("No similar documents found.")
                continue

            knowledge_context = ""

            for doc in similar_docs:
                if doc['_relevance_score'] > 0.7:
                    knowledge_context += doc['text'] + "\n"
                    

            prompt = f"""            You are a helpful assistant. Here is the context from the knowledge base:
            {knowledge_context}
            Please answer the following question based on the context: {query}"""
            print(f"Prompt for RAG Agent: {prompt}")
            response = await rag_agent.run(
                # ctx=RunContext(deps=knowledge_query),
                # message_history=message_history,
                prompt
            )

            print(f"Response: {response.output}")

            message_history = response.all_messages()
            #

# %%
if __name__ == "__main__":
    asyncio.run(main())
    print("RAG Agent setup complete.")
# %%
