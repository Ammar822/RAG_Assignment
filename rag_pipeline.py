import os
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.vectorstores.faiss import FAISS

# Load environment variables from .env
load_dotenv()

def setup_rag_chain(vector_store):
    """
    Sets up the RAG chain using FAISS retriever and Groq-hosted LLM.
    """
    if not vector_store:
        print("Vector store not initialized.")
        return None

    api_key = os.getenv("GROQ_API_KEY")
    base_url = os.getenv("BASE_URL", "https://api.groq.com/openai/v1")
    model_name = os.getenv("LLM_MODEL", "llama3-8b-8192")
    max_retrievals = int(os.getenv("MAX_RETRIEVALS", "5"))

    try:
        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0
        )
    except Exception as e:
        print(f"Error initializing ChatOpenAI: {e}")
        return None

    # Create retriever from vector store
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": max_retrievals}
    )

    # Build RAG chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )

    return rag_chain

def run_query(rag_chain, query_text):
    """
    Run a query through the RAG chain and return the answer and source docs.
    """
    if not rag_chain:
        return "RAG chain not initialized."

    try:
        result = rag_chain.invoke({"query": query_text})
        return result
    except Exception as e:
        return f"Error during query execution: {e}"
