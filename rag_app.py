import os
import uuid
import glob
from typing import List
import gradio as gr

# Setup environment variables for LM Studio
# By default LM Studio runs on localhost:1234
os.environ["OPENAI_API_KEY"] = "lm-studio"
os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1"

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import pickle
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    try:
        from langchain.retrievers.ensemble import EnsembleRetriever
    except ImportError:
        # Fallback for newer or alternative langchain split packages
        from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.stores import InMemoryStore

# For Reranking
from sentence_transformers import CrossEncoder

class AdvancedRAGSystem:
    def __init__(self, persist_directory: str = "./db", collection_name: str = "advanced_rag_collection"):
        # 1. Initialize LLMs from LM Studio
        # Adjust temperature based on your needs
        self.default_model = "qwen3-vl-30b-a3b-instruct"
        # LM Studio에서 별도의 임베딩 모델을 로드하지 않아도 되도록 로컬 허깅페이스 임베딩 사용
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # 2. Setup Vector Store (Chroma) for semantic search
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.vectorstore = self._init_chroma()

        # 3. Setup Keyword Store (BM25) for lexical search
        self.keyword_retriever = self._load_bm25()

        # 4. Setup Document Store (InMemoryStore) to hold the original parent chunks
        # In a real production app, this should be a persistent ByteStore like LocalFileStore
        self.docstore = InMemoryStore()
        self._load_docstore()

        # 5. Initialize the Reranker Model
        # This runs locally using sentence-transformers
        print("Loading CrossEncoder model for reranking...")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("Model loaded.")

        # Parent ID key metadata
        self.id_key = "parent_doc_id"
        
    def _init_chroma(self):
        os.makedirs(self.persist_directory, exist_ok=True)
        return Chroma(
            collection_name=self.collection_name, 
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        
    def _bm25_path(self):
        return os.path.join(self.persist_directory, f"{self.collection_name}_bm25.pkl")
        
    def _docstore_path(self):
        return os.path.join(self.persist_directory, f"{self.collection_name}_docstore.pkl")
        
    def _load_bm25(self):
        path = self._bm25_path()
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None
        
    def _load_docstore(self):
        path = self._docstore_path()
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.docstore.mset(data)
                
    def _save_bm25(self):
        if self.keyword_retriever:
            with open(self._bm25_path(), 'wb') as f:
                pickle.dump(self.keyword_retriever, f)
                
    def _save_docstore(self):
        # InMemoryStore doesn't expose all items easily, we'll extract them using a hack or just tracking it
        # Real-world use LocalFileStore instead
        data = list(self.docstore.yield_keys())
        # We would need a custom way to serialize docstore, skipping for simplicity if UI only needs 1 session, 
        # but since user wants persistent DB we'll attempt basic pickle if possible.
        items = [(k, self.docstore.mget([k])[0]) for k in data]
        with open(self._docstore_path(), 'wb') as f:
            pickle.dump(items, f)

    def change_db(self, collection_name: str):
        self.collection_name = collection_name
        self.vectorstore = self._init_chroma()
        self.keyword_retriever = self._load_bm25()
        self.docstore = InMemoryStore()
        self._load_docstore()
        return True

    def delete_db(self, collection_name: str):
        import chromadb
        try:
            client = chromadb.PersistentClient(path=self.persist_directory)
            client.delete_collection(name=collection_name)
        except Exception:
            pass
            
        bm25_curr = os.path.join(self.persist_directory, f"{collection_name}_bm25.pkl")
        doc_curr = os.path.join(self.persist_directory, f"{collection_name}_docstore.pkl")
        if os.path.exists(bm25_curr): os.remove(bm25_curr)
        if os.path.exists(doc_curr): os.remove(doc_curr)
        
        if self.collection_name == collection_name:
            self.change_db("advanced_rag_collection")

    # ==========================================
    # Step 1: Data Indexing (Storage Pipeline)
    # ==========================================
    def index_documents(self, documents: List[Document], parent_chunk_size: int = 1500, child_chunk_size: int = 400, model_name: str = None, progress=gr.Progress()):
        llm = ChatOpenAI(temperature=0.1, model=model_name or self.default_model)
        
        progress(0.1, desc="Splitting Parent Chunks...")
        print(f"Starting indexing for {len(documents)} document(s)...")

        # Split into Parent Chunks (Original large chunks)
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=150)
        parent_docs = parent_splitter.split_documents(documents)
        print(f"Created {len(parent_docs)} parent chunks.")

        # Generate UUIDs for parent chunks
        parent_ids = [str(uuid.uuid4()) for _ in parent_docs]

        # Add Parent Chunks to Document Store mapping doc_id -> doc
        progress(0.2, desc="Saving Parent Chunks to Document Store...")
        docstore_data = list(zip(parent_ids, parent_docs))
        self.docstore.mset(docstore_data)
        self._save_docstore()

        # We will create summaries (Child Chunks) and add them to Vector Store and BM25 index
        child_chunks = []

        print("Generating summaries for vector store indexing...")
        progress(0.3, desc="Generating LLM Summaries for Child Chunks...")
        
        # Simple prompt to summarize the chunk for better retrieval
        summary_prompt = PromptTemplate.from_template(
            "Summarize the following text concisely to capture its core meaning. This summary will be used for search retrieval.\n\nText: {text}\n\nSummary:"
        )

        for i, doc in progress.tqdm(enumerate(parent_docs), total=len(parent_docs), desc="Summarizing chunks"):
            if (i+1) % 5 == 0:
                print(f"Summarizing chunk {i+1}/{len(parent_docs)}...")
            
            # Generate summary via LLM
            # (Note for production: You can batch this process if LM studio supports it)
            chain = summary_prompt | llm | StrOutputParser()
            summary = chain.invoke({"text": doc.page_content})
            
            # Create a new document with the summary and link it back to the parent using metadata ID
            child_doc = Document(
                page_content=summary, 
                metadata={self.id_key: parent_ids[i]}
            )
            # You can also keep the original metadata (e.g., source file name)
            child_doc.metadata.update(doc.metadata)

            child_chunks.append(child_doc)

        print(f"Adding {len(child_chunks)} summary chunks to Vector Store (Chroma)...")
        progress(0.8, desc="Adding embeddings to Chroma DB...")
        self.vectorstore.add_documents(child_chunks)

        print("Building Keyword Store (BM25) index...")
        progress(0.9, desc="Building BM25 Keyword Index...")
        self.keyword_retriever = BM25Retriever.from_documents(child_chunks)
        self._save_bm25()
        
        progress(1.0, desc="Indexing completed!")

    # ==========================================
    # Step 2 & 3: Hybrid Search & Reranking
    # ==========================================
    def retrieve_and_rerank(self, query: str, top_k: int = 3) -> List[Document]:
        if self.keyword_retriever is None:
            raise ValueError("You must index documents before searching.")

        # 2a: Hybrid Search (Vector + Keyword)
        vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # EnsembleRetriever combines results and applies Reciprocal Rank Fusion (RRF)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, self.keyword_retriever],
            weights=[0.5, 0.5] # Give equal weight to keyword vs semantic
        )
        
        print(f"Performing Hybrid search for query: '{query}'...")
        # Get Candidate summaries
        candidate_summaries = ensemble_retriever.invoke(query)
        
        # Extract unique parent IDs from candidate summaries
        parent_ids = list(set([doc.metadata[self.id_key] for doc in candidate_summaries if self.id_key in doc.metadata]))
        
        # 3a: Small-to-Big Retrieval
        # Fetch actual original parent chunks from the Document Store
        parent_docs = []
        for doc_id, doc in zip(parent_ids, self.docstore.mget(parent_ids)):
            if doc:
                parent_docs.append(doc)
                
        print(f"Retrieved {len(parent_docs)} parent chunks. Reranking...")

        # 3b: Reranking
        # We score each query/parent-doc pair using the cross encoder
        pairs = [[query, doc.page_content] for doc in parent_docs]
        scores = self.cross_encoder.predict(pairs)

        # Sort documents by score descending
        scored_docs = list(zip(scores, parent_docs))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Return the top K parent documents
        best_docs = [doc for score, doc in scored_docs[:top_k]]
        return best_docs

    # ==========================================
    # Step 4: Final LLM Generation
    # ==========================================
    def generate_answer(self, query: str, model_name: str = None) -> str:
        llm = ChatOpenAI(temperature=0.1, model=model_name or self.default_model)
        # Retrieve context (Parent chunks)
        context_docs = self.retrieve_and_rerank(query, top_k=2)
        
        # Format the context texts into a single string
        context_str = "\n\n".join([f"--- Document ---\n{doc.page_content}" for doc in context_docs])
        
        # Prompt template combining context and question
        qa_prompt = PromptTemplate.from_template(
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )

        # Generation pipeline using LCEL
        chain = qa_prompt | llm | StrOutputParser()
        
        print("\n=== Generating final answer via LLM ===")
        # Invoke the chain
        response = chain.invoke({"context": context_str, "question": query})
        return response

import gradio as gr

def get_available_collections(persist_directory="./db"):
    os.makedirs(persist_directory, exist_ok=True)
    # In ChromaDB we can infer DB presence via the sqlite file
    if os.path.exists(os.path.join(persist_directory, "chroma.sqlite3")):
        import chromadb
        client = chromadb.PersistentClient(path=persist_directory)
        cols = client.list_collections()
        if cols:
            return [c.name for c in cols]
    return ["advanced_rag_collection"]

def main():
    rag = AdvancedRAGSystem()

    def index_files(files, collection_name, model_name, progress=gr.Progress()):
        if not files:
            return "Please provide some files to index.", gr.update()
        
        try:
            rag.change_db(collection_name)
            documents = []
            progress(0.01, desc="Extracting text from files...")
            for file_path in files:
                if file_path.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                elif file_path.endswith('.txt'):
                    # Some TXTs might not be purely utf-8, handle gracefully or use autodetect
                    loader = TextLoader(file_path, encoding='utf-8')
                    try:
                        documents.extend(loader.load())
                    except Exception:
                        loader = TextLoader(file_path)
                        documents.extend(loader.load())
            
            if not documents:
                return "Failed to extract documents from files.", gr.update()
                
            rag.index_documents(documents, parent_chunk_size=300, child_chunk_size=100, model_name=model_name, progress=progress)
            
            # Refresh collection list
            cols = get_available_collections()
            return f"Indexing complete! {len(documents)} chunks loaded into DB '{collection_name}'.", gr.update(choices=cols, value=collection_name)
        except Exception as e:
            # Check for specifically LM Studio load errors
            if "No models loaded" in str(e):
                return "ERROR: LM Studio has no model loaded. Please click 'Load Model' in LM Studio first!", gr.update()
            return f"Error during indexing: {str(e)}", gr.update()

    def chat_interface(message, history, collection_name, model_name):
        try:
            rag.change_db(collection_name)
            answer = rag.generate_answer(message, model_name=model_name)
            return answer
        except ValueError as e:
            return str(e)  # Catch "You must index documents before searching."
        except Exception as e:
            if "No models loaded" in str(e):
                return "ERROR: LM Studio has no model loaded. Please click 'Load Model' in LM Studio first!"
            return f"An error occurred: {str(e)}"

    # Build Gradio UI
    with gr.Blocks(title="Advanced RAG Pipeline") as demo:
        gr.Markdown("# 🚀 Advanced RAG Architecture using LangChain & LM Studio")
        
        with gr.Row():
            model_input = gr.Textbox(value="qwen3-vl-30b-a3b-instruct", label="LM Studio Model Name")
            
        with gr.Row():
            cols = get_available_collections()
            collection_dropdown = gr.Dropdown(choices=cols, value="advanced_rag_collection", label="Select DB Collection", allow_custom_value=True)
            
            new_db_input = gr.Textbox(label="New DB Name", placeholder="Type name and click Add...")
            add_db_btn = gr.Button("➕ Add DB", variant="secondary")
            delete_db_btn = gr.Button("🗑️ Delete DB", variant="stop")
            
        def add_db_ui(new_name):
            if not new_name:
                return gr.update(), gr.update()
            rag.change_db(new_name)
            cols = get_available_collections()
            if new_name not in cols:
                cols.append(new_name)
            return gr.update(choices=cols, value=new_name), ""
            
        def delete_db_ui(del_name):
            rag.delete_db(del_name)
            cols = get_available_collections()
            val = cols[0] if cols else "advanced_rag_collection"
            if not cols: cols = [val]
            rag.change_db(val)
            return gr.update(choices=cols, value=val)
            
        add_db_btn.click(fn=add_db_ui, inputs=[new_db_input], outputs=[collection_dropdown, new_db_input])
        delete_db_btn.click(fn=delete_db_ui, inputs=[collection_dropdown], outputs=[collection_dropdown])
            
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Index Data")
                file_input = gr.File(file_count="multiple", file_types=[".txt", ".pdf"], label="Drag and drop or upload PDF/TXT files")
                index_btn = gr.Button("Index Data", variant="primary")
                index_status = gr.Textbox(label="Status", interactive=False)
                
                index_btn.click(
                    fn=index_files, 
                    inputs=[file_input, collection_dropdown, model_input], 
                    outputs=[index_status, collection_dropdown]
                )
                
            with gr.Column(scale=2):
                gr.Markdown("### 2. Chat with RAG")
                gr.ChatInterface(fn=chat_interface, additional_inputs=[collection_dropdown, model_input])
                
    # Launch on localhost
    demo.launch(server_name="127.0.0.1", server_port=7860)

if __name__ == "__main__":
    main()
