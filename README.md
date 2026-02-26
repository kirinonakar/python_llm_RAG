# Advanced RAG System

An advanced Retrieval-Augmented Generation (RAG) pipeline implemented in Python using LangChain, ChromaDB, and LM Studio. It features a robust architecture to improve retrieval accuracy and a clean, interactive Gradio frontend for indexing and chatting.

## 🚀 Key Features

*   **Hybrid Search**: Combines semantic search (ChromaDB) and lexical/keyword search (BM25) using LangChain's `EnsembleRetriever`.
*   **Reciprocal Rank Fusion (RRF)**: Implements RRF to optimally merge and rank results from the dual stores.
*   **Small-to-Big Retrieval**: Initially retrieves small, highly-relevant "Child Chunk" summaries, and then fetches the connected large "Parent Chunks" to provide rich context to the LLM.
*   **Local Reranking**: Uses `sentence-transformers` (Cross-Encoder) natively to re-score the fetched parent chunks against the user's query for maximum accuracy.
*   **LM Studio Integration**: Designed to be entirely local by connecting to an LM Studio server hosting your local models (OpenAI compatible API).
*   **Interactive Gradio UI**: Readily drag-and-drop `.pdf` or `.txt` files to index, manage multiple databases dynamically, and chat with your indexed documents.
*   **Persistent Storage**: Automatically saves vector embeddings and keyword indexes locally securely.

## 🛠️ Prerequisites

*   Python 3.10+
*   [LM Studio](https://lmstudio.ai/) installed and running locally with the Local Server turned on (default: `http://localhost:1234/v1`).
*   A Chat/Instruct Model loaded in LM Studio. (e.g., `qwen3-vl-30b-a3b-instruct`).

## ⚙️ Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/advanced-rag-system.git
    cd advanced-rag-system
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv .venv
    
    # Windows
    source .venv/Scripts/activate
    
    # MacOS/Linux
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♂️ Usage

1.  Ensure your LM Studio server is running locally. Make sure the model loaded matches the name you plan to use in the UI.

2.  Run the application:
    ```bash
    python rag_app.py
    ```

3.  Open the provided URL locally in your browser (usually `http://127.0.0.1:7860`).

### Using the UI

*   **Model Name**: At the top left, ensure the LM Studio Model Name matches exactly what is loaded in LM Studio.
*   **Database Management**: Use the dropdown to select a Vector Database, or use the *New DB Name* text box to create a brand new isolated database. You can also delete databases using the **Delete DB** button.
*   **Indexing Documents**: Upload multiple PDFs or TXT files via the file uploader. Click **Index Data** to split, process, embed, and store the documents in your selected database. A progress bar will track the process.
*   **Chat**: Ask the ChatInterface questions about your documents in the selected database!

## 📦 Dependencies

*   `langchain`
*   `langchain-community`, `langchain-openai`, `langchain-chroma`
*   `chromadb`
*   `rank_bm25`
*   `sentence-transformers`
*   `tiktoken`
*   `gradio`
*   `pypdf`

## 📄 License
This project is distributed under the `MIT License`. See [LICENSE](LICENSE) for more information.
