import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def parse_responses(resp_str):
    if not isinstance(resp_str, str):
        return []
    return [r.strip().lower() for r in resp_str.split("|")]


def jaccard_similarity(a, b):
    set_a = set(a)
    set_b = set(b)
    
    if not set_a and not set_b:
        return 1.0
    
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    return intersection / union if union > 0 else 0


def compute_score(source_row, target_row, desc_sim):
    # 1. field_id match
    field_id_match = 1.0 if source_row["field_id"] == target_row["field_id"] else 0.0

    # 2. response similarity
    source_resp = parse_responses(source_row["values"])
    target_resp = parse_responses(target_row["values"])
    
    response_sim = jaccard_similarity(source_resp, target_resp)

    # final score
    score = (
        0.4 * field_id_match +
        0.4 * desc_sim +
        0.2 * response_sim
    )

    return score, field_id_match, response_sim


def get_top_matches(source_row, source_emb, target_df, target_embs, top_k=5):
    desc_sims = cosine_similarity([source_emb], target_embs)[0]
    
    results = []
    
    for i, target_row in target_df.iterrows():
        score, field_match, resp_sim = compute_score(
            source_row,
            target_row,
            desc_sims[i]
        )
        
        results.append({
            "target_field": target_row["field_id"],
            "description": target_row["field_desc"],
            "score": float(score),
            "desc_sim": float(desc_sims[i]),
            "field_match": field_match,
            "response_sim": resp_sim
        })
    
    # sort by final score
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    
    return results[:top_k]