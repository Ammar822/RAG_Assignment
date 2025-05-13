# RAG_Assignment
Here's a well-structured and personalized `README.md` file based on your work and preferences:

---

````markdown
# 🧠 RAG System - Assignment Submission

This repository contains my implementation of a **Retrieval-Augmented Generation (RAG)** system for answering user queries based on a set of documents. The system is modular, allowing experimentation with different retrieval strategies, document chunking, and evaluation metrics.

---

## ⚙️ Setup Instructions

To run this project on your machine:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/RAG_Assignment.git
   cd RAG_Assignment
````

2. **Create a virtual environment and activate it:**

   ```bash
   conda create -n RAG_env python=3.10
   conda activate RAG_env
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Add a `.env` file** with your settings:

   ```env
   GROQ_API_KEY=your_groq_api_key
   BASE_URL=https://api.groq.com/openai/v1
   LLM_MODEL=llama3-8b-8192
   ```

5. **Run the test script to evaluate the system:**

   ```bash
   python test_rag_pipeline.py
   ```

---

## 🏗️ RAG System Architecture

* **File Structure:**

  ```
  ├── documents/              # Input source PDFs
  ├── vector_store/           # FAISS vector store
  ├── rag_pipeline.py         # Core RAG pipeline logic
  ├── evaluation.py           # Answer and retrieval evaluation
  ├── test_rag_pipeline.py    # Testing with predefined questions
  └── .env                    # Environment configuration
  ```

* **Pipeline Components:**

  * **Document Loader:** Loads PDFs and splits into chunks using LangChain.
  * **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
  * **Vector Store:** FAISS for fast similarity search.
  * **LLM:** Accessed via Groq API, using the LLaMA 3 model.
  * **Retriever:** Returns top-k relevant chunks.
  * **Generator:** Uses `RetrievalQA` to answer based on context.

---

## 🔍 Retrieval Strategy Experiments

| Strategy               | Chunk Size       | Top-k | Notes                                 |
| ---------------------- | ---------------- | ----- | ------------------------------------- |
| Sentence-based chunks  | 200              | 5     | Best performance overall              |
| Paragraph-level chunks | 500              | 3     | High precision, slightly lower recall |
| Overlapping chunks     | 250 (50 overlap) | 5     | Improved context, slower indexing     |

---

## 📊 Evaluation Metrics

| Test Case                         | Precision | Recall | F1 Score | Similarity Score |
| --------------------------------- | --------- | ------ | -------- | ---------------- |
| GPT-3 Contribution                | 0.0       | 0.0    | 0.0      | 0.37             |
| Advantages of Vision Transformers | 0.5       | 1.0    | 0.67     | 0.62             |
| BERT Pretraining Objective        | 0.5       | 1.0    | 0.67     | 0.73             |

* **Average Retrieval Precision:** 0.33
* **Average Answer Similarity:** 0.57

Evaluation used:

* Binary relevance for document retrieval
* Semantic similarity via `SequenceMatcher` for answer quality

---

## ✅ Strengths & Weaknesses

**Strengths:**

* Modular pipeline with clean separation of concerns
* Works seamlessly with HuggingFace and FAISS
* Quick experimentation with prompt templates and chunk sizes
* Good semantic match even when exact answer isn’t retrieved

**Weaknesses:**

* Answer generation can hallucinate if retrieval is weak
* Over-retrieval of duplicates from same document
* Evaluation based on string similarity is limited for nuance

---

## 🛠️ Challenges Faced & Solutions

| Challenge                                 | Solution                                                        |
| ----------------------------------------- | --------------------------------------------------------------- |
| Deprecation warnings in LangChain         | Updated imports to `langchain_community`                        |
| Repeated document retrieval               | Used `set()` to remove duplicates before evaluation             |
| Ambiguity in ground-truth answers         | Used rough semantic similarity instead of strict matching       |
| Tokenizer fork warnings from HuggingFace  | Disabled parallelism for tokenizers during execution            |
| Testing different configurations manually | Wrote a flexible `test_rag_pipeline.py` to easily plug in cases |

---

## 📁 Final Notes

This project demonstrates how modern RAG systems can be built, evaluated, and improved using open-source tools and real-world documents. It’s a complete end-to-end implementation with evaluation and testing.

Feel free to fork or adapt for your own experiments.

```

---
```
