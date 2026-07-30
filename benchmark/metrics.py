import math


def precision_at_k(recommended, relevant, k):
    top_k = recommended[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & relevant) / len(top_k)


def recall_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    return len(set(top_k) & relevant) / len(relevant)


def hit_rate_at_k(recommended, relevant, k):
    top_k = recommended[:k]
    return 1.0 if set(top_k) & relevant else 0.0


def ndcg_at_k(recommended, relevant, k):
    top_k = recommended[:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, item in enumerate(top_k) if item in relevant)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0
