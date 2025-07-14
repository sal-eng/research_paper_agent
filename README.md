# Research Paper QnA Bot


This project implements a **Retrieval Augmented Generation (RAG) Agentic system** designed to answer user queries by leveraging a **knowledge base stored in LanceDB**. It uses an AI agent to generate a relevant query for the knowledge base, retrieves similar documents, and then uses another AI agent to synthesize an answer based on the retrieved context. This approach ensures more accurate and contextually rich responses compared to standalone generative models.

---

## Features

* **LanceDB Integration:** Utilizes **LanceDB** for efficient storage and retrieval of document embeddings, enabling fast similarity searches.
* **Ollama Embedding Model:** Employs the **`nomic-embed-text` model from Ollama** for generating text embeddings.
* **Text Chunking:** Includes a robust text chunking mechanism using `tiktoken` to handle large documents effectively.
* **Hybrid Search and Reranking:** Implements **hybrid search** (vector similarity + full-text search) with a `LinearCombinationReranker` for improved retrieval relevance.
* **AI-Powered Query Generation:** An initial AI agent (**`Knowledge Query Agent`**) analyzes user input to formulate precise queries for the knowledge base.
* **AI-Powered Response Generation:** A main RAG agent (**`RAG Agent`**) synthesizes answers based on the retrieved documents, ensuring contextually informed responses.
* **Configurable Parameters:** Allows customization of chunk size, retrieval limits, and reranker weights.

---

## Installation

To set up and run this project, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install `uv` (if you haven't already):**

    If you don't have `uv` installed, the recommended way is via its standalone installer for the best performance and isolation:

    ```bash
    curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
    ```
    (For Windows, see the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)).

3.  **Create and activate a virtual environment:**

    `uv` provides blazing-fast virtual environment management. This will create a `.venv` directory in your project root.

    ```bash
    uv venv
    # Activate the virtual environment:
    # On Windows
    .\.venv\Scripts\activate
    # On macOS/Linux
    source ./.venv/bin/activate
    ```

4.  **Install the required Python packages:**

    `uv` uses a `uv.lock` file (generated from `pyproject.toml`) for exact dependency reproducibility.

    
    ```bash
    uv sync
    ```
        

5.  **Set up your environment variables:**
    Create a `.env` file in the project root and add your API keys for the Gemini model. For example:

    ```
    GOOGLE_API_KEY=your_gemini_api_key
    ```
    Ensure that you have **Ollama running** and the **`nomic-embed-text` model pulled**:
    ```bash
    ollama pull nomic-embed-text
    ```

6.  **Prepare your knowledge base:**
    Create a directory named `knowledge_markdowns` in the project root and place your `.md` (Markdown) files inside it. These files will be used to populate the LanceDB knowledge base.

---

## Usage

1.  **Set up the LanceDB knowledge base:**
    Run the `lancedb_setup.py` script once to initialize the LanceDB database and populate it with your markdown documents. This script will create a `lance_db` directory and a table named `knowledge`.

    ```bash
    python lancedb_setup.py
    ```

2.  **Run the RAG agent:**
    Execute the `rag_agent.py` script to start the interactive RAG system.

    ```bash
    python rag_agent.py
    ```

3.  **Interact with the RAG agent:**
    Once the `rag_agent.py` script is running, you'll be prompted to enter your queries. The system will then generate a knowledge query, retrieve relevant documents, and provide an answer based on the retrieved context.

    ```
    Enter your query (or 'exit' to quit): What is the attention mechanism?
    Generated Knowledge Query: attention mechanism
    Prompt for RAG Agent: You are a helpful assistant. Here is the context from the knowledge base:
    ... (retrieved document content) ...
    Please answer the following question based on the context: What is the attention mechanism?
    Response: The attention mechanism in deep learning allows models to focus on relevant parts of the input sequence when processing data, ...
    Enter your query (or 'exit' to quit): exit
    ```

---

## Project Structure

* **`lancedb_setup.py`**: Handles **LanceDB initialization**, schema definition, text chunking, document ingestion, and retrieval functions. It creates and manages the `knowledge` table.
* **`load_models.py`**: Loads and configures the **Gemini AI model** used by the agents.
* **`rag_agent.py`**: Contains the core logic for the RAG system, including the `Knowledge Query Agent` and the `RAG Agent`. It orchestrates the query generation, document retrieval, and response synthesis process.
* **`pyproject.toml`**: For uv dependency management
* **`uv.lock`**: Generated by uv for locking the dependencies

* **`.env`**: Stores environment variables, such as API keys.
* **`knowledge_markdowns/`**: A directory where you place your markdown files to be used as the knowledge base.

---

## API

The project exposes the following key functions and classes for interacting with the LanceDB and AI agents:

### `lancedb_setup.py`

* **`class Document(LanceModel)`**: Defines the LanceDB schema for documents, including `id`, `text`, and `vector` fields.
* **`chunk_text(text: str, max_tokens: int = 1000, encoding_name: str = "cl100k_base") -> list[str]`**: Chunks input text into smaller pieces based on token limits.
* **`create_documents_table(db_path: str, table_name: str, overwrite: bool = True) -> LanceTable`**: Connects to LanceDB and creates a new table or overwrites an existing one.
* **`delete_documents_table(db_path: str, table_name: str) -> None`**: Deletes a specified LanceDB table.
* **`add_docs_to_table(table: LanceTable, docs_dir: str, max_tokens: int = 8192) -> None`**: Reads markdown files from a directory, chunks them, and adds them to a LanceDB table.
* **`retrieve_similar_docs(table: LanceTable, query: str, query_type: str = "hybrid", limit: int = 100, reranker_weight: float = 0.7) -> list[dict[str, Any]]`**: Performs a hybrid search on the LanceDB table to retrieve documents similar to the given query, applying a reranker.
* **`setup_lancedb() -> None`**: The main setup function to create and populate the LanceDB.

### `load_models.py`

* **`GEMINI_MODEL: GeminiModel`**: An instance of the Gemini AI model, initialized with `gemini-2.5-flash-preview-04-17` and `google-gla` provider.

### `rag_agent.py`

* **`setup_knowledge_query_agent() -> Agent`**: Configures and returns an `Agent` instance responsible for generating a knowledge base query from user input.
* **`setup_rag_agent() -> Agent`**: Configures and returns an `Agent` instance responsible for generating a response based on provided context and user query.
* **`main()`**: The asynchronous main function that orchestrates the RAG workflow, handling user input, agent interactions, and output.

---

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix: `git checkout -b feature/your-feature-name` or `bugfix/fix-something`.
3.  Make your changes and ensure tests pass (if applicable).
4.  Commit your changes: `git commit -m "feat: Add new feature"`.
5.  Push to your branch: `git push origin feature/your-feature-name`.
6.  Open a pull request to the `main` branch of this repository.

---

## License

This project is licensed under the **MIT License**.

---

## Contact/Support

For questions or support, please open an issue in the GitHub repository.

---

## Acknowledgments

* [LanceDB](https://lancedb.com/) for the vector database solution.
* [Ollama](https://ollama.ai/) for local large language models.
* [Google Gemini](https://ai.google.dev/models/gemini) for the powerful generative AI models.
* [tiktoken](https://github.com/openai/tiktoken) for efficient tokenization.