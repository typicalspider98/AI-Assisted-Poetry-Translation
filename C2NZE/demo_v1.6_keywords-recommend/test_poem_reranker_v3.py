import json 
from semantic_helper import redis_dict, search_topk_similar_batch, extract_base_word
from FlagEmbedding import FlagReranker

import numpy as np

def custom_sigmoid(x, temperature=1.5):
    return 1 / (1 + np.exp(-temperature * x))

# Initialize reranker (bge-reranker-v2-m3, non-prompt model)
reranker = FlagReranker("../models/bge-reranker-v2-m3", use_fp16=True)

def rerank_related_to_keywords(keywords, top_k=5, model_id=2):
    for kw in keywords:
        print(f"\n🔍 Keyword: {kw}")
        topk = search_topk_similar_batch([kw], top_k=top_k, model_id=model_id)

        pairs = []
        passages = []
        redis_keys = []

        for key, cos_sim in topk:
            word = extract_base_word(key)
            meaning = "(no explanation)"

            if redis_dict.exists(key):
                try:
                    entry = json.loads(redis_dict.get(key))
                    meaning = entry.get("meaning", meaning)
                except:
                    pass

            # ✅ Pure semantic pair, no prompt added
            passage = f"{word}: {meaning}"
            pairs.append([kw, passage])
            passages.append(passage)
            redis_keys.append(key)

        print("📘 TopK retrieved (by vector similarity):")
        for i, (rk, cs, ps) in enumerate(zip(redis_keys, [s for _, s in topk], passages)):
            print(f"{i+1}. {rk:<30} | CosSim: {cs:.4f} | Content: {ps}")

        print("📊 Reranker similarity scores (standard semantic input):")
        scores = reranker.compute_score(pairs, normalize=False)
        adjusted_scores = [custom_sigmoid(x, temperature=2.0) for x in scores]
        for i, (pair, score) in enumerate(zip(pairs, adjusted_scores)):
            print(f"{i+1}. [{pair[0]}] → {pair[1]:<50s} | RerankScore: {score:.4f}")

if __name__ == "__main__":
    # You can replace with keywords extracted by LLM
    keywords = ["月","moon", "light", "bed", "hometown", "sorrow", "frost", "ground", "望"]
    rerank_related_to_keywords(keywords, top_k=5)

    # Explicitly stop the thread pool (though internal behavior may vary)
    try:
        reranker.stop_self_pool()
    except Exception as e:
        print(f"[Warning] stop_self_pool failed: {e}")

    # Explicitly release the reranker object to avoid re-triggering __del__
    del reranker
