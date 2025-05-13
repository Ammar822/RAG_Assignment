from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_retrieval(expected_docs, retrieved_docs, use_fuzzy=False, threshold=0.8):
    expected_set = set(expected_docs)
    retrieved_set = set(retrieved_docs)

    if use_fuzzy:
        def fuzzy_match(doc, doc_set):
            return any(SequenceMatcher(None, doc.lower(), other.lower()).ratio() > threshold for other in doc_set)
        true_positives = sum(fuzzy_match(doc, expected_set) for doc in retrieved_set)
    else:
        true_positives = len(expected_set & retrieved_set)

    false_positives = len(retrieved_set) - true_positives
    false_negatives = len(expected_set) - true_positives

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1": round(f1, 2)
    }

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def evaluate_answer_quality(expected_answer, generated_answer):
    embeddings = model.encode([expected_answer, generated_answer], convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return round(similarity_score, 2)

