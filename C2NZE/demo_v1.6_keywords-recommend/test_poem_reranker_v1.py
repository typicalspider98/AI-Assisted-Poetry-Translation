import json
from semantic_helper import redis_dict, search_topk_similar_batch, extract_base_word
from FlagEmbedding import FlagReranker

# 初始化 reranker（使用多语言版本）
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

            # ✅ 使用：仅 单词 + 释义，不含例句
            passage = f"{word}: {meaning}"
            pairs.append([kw, passage])
            passages.append(passage)
            redis_keys.append(key)

        print("📘 TopK 召回（按向量相似度）:")
        for i, (rk, cs, ps) in enumerate(zip(redis_keys, [s for _, s in topk], passages)):
            print(f"{i+1}. {rk:<30} | CosSim: {cs:.4f} | 内容: {ps}")

        print("📊 reranker 相似度打分:")
        scores = reranker.compute_score(pairs, normalize=True)
        for i, (pair, score) in enumerate(zip(pairs, scores)):
            print(f"{i+1}. Passage = {pair[1]:<50s} | RerankScore: {score:.4f}")

if __name__ == "__main__":
    # ✅ 示例关键词，可替换为 LLM 输出或人工指定
    keywords = ["moon", "light", "bed", "home", "sorrow"]
    rerank_related_to_keywords(keywords, top_k=5)

    
# 🔚 显式释放线程池，避免 __del__ 触发异常
if reranker and hasattr(reranker, "stop_self_pool") and callable(reranker.stop_self_pool):
    reranker.stop_self_pool()