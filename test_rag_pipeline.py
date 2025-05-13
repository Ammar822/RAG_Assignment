from rag_pipeline import setup_rag_chain, run_query
from document_processor import load_vector_store
from evaluation import evaluate_answer_quality, evaluate_retrieval
from langchain_community.embeddings import HuggingFaceEmbeddings

# Define test queries and expected outputs
test_cases = [
    {
        "query": "What is the main contribution of GPT-3?",
        "expected_answer": "GPT-3 demonstrated that large language models can perform few-shot learning without fine-tuning.",
        "expected_sources": ["gpt3_language_models_are_few_shot_learners.pdf"]
    },
    {
        "query": "What are the advantages of Vision Transformers?",
        "expected_answer": "Vision Transformers perform better than CNNs on large-scale image classification tasks.",
        "expected_sources": ["vision_transformers_at_scale.pdf"]
    },
    {
        "query": "What is BERT pretraining objective?",
        "expected_answer": "BERT uses masked language modeling and next sentence prediction as pretraining objectives.",
        "expected_sources": ["bert_pretraining.pdf"]
    }
]

def main():
    print("\n🚀 Initializing RAG pipeline...\n")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = load_vector_store("vector_store", embeddings)
    rag_chain = setup_rag_chain(vector_store)

    if not rag_chain:
        print("❌ RAG chain setup failed.")
        return

    for i, case in enumerate(test_cases):
        print(f"\n==============================")
        print(f"🔍 Test Case {i+1}: {case['query']}")
        print(f"==============================")

        result = run_query(rag_chain, case["query"])

        # Extract results
        generated_answer = result.get("result", "")
        retrieved_docs = [doc.metadata.get("source", "").split("/")[-1] for doc in result.get("source_documents", [])]

        # Evaluate
        answer_score = evaluate_answer_quality(case["expected_answer"], generated_answer)
        retrieval_scores = evaluate_retrieval(case["expected_sources"], retrieved_docs)

        # Print results
        print(f"\n📥 Generated Answer:\n{generated_answer}")
        print(f"\n📄 Retrieved Documents:\n{retrieved_docs}")
        print(f"\n📊 Answer Similarity Score: {answer_score}")
        print(f"📈 Retrieval Evaluation: {retrieval_scores}")

if __name__ == "__main__":
    main()
