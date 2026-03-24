import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_top_matches(source_emb, target_emb, target_df, top_k=5):
    sims = cosine_similarity([source_emb], target_emb)[0]

    top_idx = np.argsort(sims)[::-1][:top_k]

    results = []
    for i in top_idx:
        results.append({
            "target_field": target_df.iloc[i]["field_id"],
            "description": target_df.iloc[i]["field_desc"],
            "score": float(sims[i])
        })
    return results
