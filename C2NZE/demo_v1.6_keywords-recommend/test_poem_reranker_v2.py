import json
from semantic_helper import redis_dict, search_topk_similar_batch, extract_base_word
from FlagEmbedding import FlagReranker


import numpy as np

def custom_sigmoid(x, temperature=1.5):
    return 1 / (1 + np.exp(-temperature * x))


# 初始化 reranker（bge-reranker-v2-m3，非 prompt 模型）
reranker = FlagReranker("../models/bge-reranker-v2-m3", use_fp16=True)

def rerank_related_to_keywords(keywords, top_k=5, model_id=2):
    for kw in keywords:
        print(f"\n🔍 关键词: {kw}")
        topk = search_topk_similar_batch([kw], top_k=top_k, model_id=model_id)

        pairs = []
        passages = []
        redis_keys = []

        for key, cos_sim in topk:
            word = extract_base_word(key)
            meaning = "(无释义)"

            if redis_dict.exists(key):
                try:
                    entry = json.loads(redis_dict.get(key))
                    meaning = entry.get("meaning", meaning)
                except:
                    pass

            # ✅ 纯语义对输入，不加 prompt
            passage = f"{word}: {meaning}"
            pairs.append([kw, passage])
            passages.append(passage)
            redis_keys.append(key)

        print("📘 TopK 召回（按向量相似度）:")
        for i, (rk, cs, ps) in enumerate(zip(redis_keys, [s for _, s in topk], passages)):
            print(f"{i+1}. {rk:<30} | CosSim: {cs:.4f} | 内容: {ps}")

        print("📊 reranker 相似度打分（标准语义输入）:")
        scores = reranker.compute_score(pairs, normalize=False)
        adjusted_scores = [custom_sigmoid(x, temperature=2.0) for x in scores]
        for i, (pair, score) in enumerate(zip(pairs, adjusted_scores)):
        # for i, (pair, score) in enumerate(zip(pairs, scores)):
            print(f"{i+1}. [{pair[0]}] → {pair[1]:<50s} | RerankScore: {score:.4f}")

if __name__ == "__main__":
    # 可替换为 LLM 提取的关键词
    keywords = ["月",  "moonlight: soft, silvery light of the moon", "frost: icy, wintry coating on the ground", "gaze: act of looking intently upward", "homeland: place of origin or birth", "nostalgia: yearning for home or the past"]
    rerank_related_to_keywords(keywords, top_k=5)

    # 显式关闭线程池（虽然它内部实现不一定稳定）
    try:
        reranker.stop_self_pool()
    except Exception as e:
        print(f"[Warning] stop_self_pool failed: {e}")

    # 显式释放 reranker 对象，避免 __del__ 再触发
    del reranker
