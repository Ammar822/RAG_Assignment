import os
from dotenv import load_dotenv

from document_loader import load_documents
from document_processor import (
    split_documents,
    create_vector_store,
    load_vector_store
)
from rag_pipeline import setup_rag_chain, run_query
from langchain_huggingface import HuggingFaceEmbeddings  


def main():
    # Load environment variables from .env
    load_dotenv()

    # Get configuration from environment
    vector_store_path = os.getenv("VECTOR_DB_PATH", "vector_store")
    embedding_model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Initialize embedding model
    print(f"Using embedding model: {embedding_model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=f"sentence-transformers/{embedding_model_name}")

    # Load vector store (if it exists), otherwise process documents
    vector_store = load_vector_store(vector_store_path, embeddings)

    if not vector_store:
        print("No existing vector store found. Loading and processing documents...")
        documents = load_documents()
        if not documents:
            print("No documents loaded. Make sure your 'documents' folder contains .pdf, .docx, or .txt files.")
            return
        chunks = split_documents(documents, chunk_size, chunk_overlap)
        vector_store = create_vector_store(chunks, embeddings, vector_store_path)

    # Set up the RAG pipeline
    rag_chain = setup_rag_chain(vector_store)
    if not rag_chain:
        print("RAG chain setup failed. Please check your Groq API key and LLM configuration.")
        return

    # Start interactive loop
    print("\n✅ RAG System initialized! Enter a query or type 'quit' to exit.")
    while True:
        query = input("\nYour query: ")
        if query.strip().lower() in ['quit', 'exit', 'q']:
            print("Exiting. Goodbye!")
            break

        result = run_query(rag_chain, query)
        print("\n" + "=" * 50)
        print("ANSWER:")
        print(result['result'] if isinstance(result, dict) else result)
        print("=" * 50)

if __name__ == "__main__":
    main()
