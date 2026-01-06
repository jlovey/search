import numpy as np

def calculate_precision_at_k(retrieved_ids, ground_truth_ids, k):
    retrieved_at_k = retrieved_ids[:k]
    intersection = set(retrieved_at_k).intersection(set(ground_truth_ids))
    return len(intersection) / k

def calculate_recall_at_k(retrieved_ids, ground_truth_ids, k):
    retrieved_at_k = retrieved_ids[:k]
    intersection = set(retrieved_at_k).intersection(set(ground_truth_ids))
    if len(ground_truth_ids) == 0:
        return 0.0
    return len(intersection) / len(ground_truth_ids)

def calculate_dcg(relevances, k):
    relevances = np.asarray(relevances, dtype=float)[:k]
    if relevances.size:
        return np.sum(relevances / np.log2(np.arange(2, relevances.size + 2)))
    return 0.0

def calculate_ndcg_at_k(retrieved_ids, ground_truth_ids, k):
    # For simplicity, assume relevance is binary (1 if in ground_truth, 0 otherwise)
    relevances = [1 if r_id in ground_truth_ids else 0 for r_id in retrieved_ids[:k]]
    actual_dcg = calculate_dcg(relevances, k)
    
    # Ideal DCG: all ground truth items are at the top
    ideal_relevances = sorted([1 if i < len(ground_truth_ids) else 0 for i in range(k)], reverse=True)
    ideal_dcg = calculate_dcg(ideal_relevances, k)
    
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg
