import json
from semantic_helper import redis_dict, search_topk_similar_batch, extract_base_word
from FlagEmbedding import FlagReranker
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import torch


def custom_sigmoid(x, temperature=1.5):
    return 1 / (1 + np.exp(-temperature * x))

# === 初始化两个 reranker ===
# [1] FlagEmbedding reranker（bge-reranker-v2-m3）
flag_reranker = FlagReranker("../../models/bge-reranker-v2-m3", use_fp16=True)

# [2] GTE multilingual reranker
model_name_or_path = "../../models/gte-multilingual-reranker-base"
gte_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
gte_model = AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path, trust_remote_code=True, torch_dtype=torch.float16
).eval().to("cuda")

def rerank_related_to_keywords(keywords, top_k=5, model_id=2):
    for kw in keywords:
        print(f"\n🔍 Keyword: {kw}")
        topk = search_topk_similar_batch([kw], top_k=top_k, model_id=model_id)

        pairs = []
        passages = []
        redis_keys = []
        cos_sims = []

        for key, cos_sim in topk:
            word = extract_base_word(key)
            meaning = "(no explanation)"

            if redis_dict.exists(key):
                try:
                    entry = json.loads(redis_dict.get(key))
                    meaning = entry.get("meaning", meaning)
                except:
                    pass

            passage = f"{word}: {meaning}"
            pairs.append([kw, passage])
            passages.append(passage)
            redis_keys.append(key)
            cos_sims.append(cos_sim)

        print("\n--- 当前pairs内容 ---")
        for p in pairs:
            print(f"左边(query): {p[0]} || 右边(passage): {p[1]}")
        print("--- 结束 ---\n")
        scores_flag = flag_reranker.compute_score(pairs, normalize=False)
        adjusted_flag = [custom_sigmoid(x, temperature=2.0) for x in scores_flag]

        with torch.no_grad():
            gte_inputs = gte_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to("cuda")
            gte_logits = gte_model(**gte_inputs).logits.view(-1).float()
            gte_scores = [custom_sigmoid(x.item(), temperature=2.0) for x in gte_logits]

        print("\n📋 Markdown Table Output:")
        print("| Rank | Redis Key | CosSim | Flag Score | GTE Score | Explanation |")
        print("|------|------------|--------|-------------|------------|-------------|")
        for i in range(len(pairs)):
            key = redis_keys[i]
            cs = cos_sims[i]
            flag = adjusted_flag[i]
            gte = gte_scores[i]
            explanation = passages[i].split(":", 1)[-1].strip()
            print(f"| {i+1} | {key} | {cs:.4f} | {flag:.4f} | {gte:.4f} | {explanation} |")

if __name__ == "__main__":
    keywords = ["月","moon", "light", "bed", "hometown", "sorrow", "frost", "ground", "望"]
    rerank_related_to_keywords(keywords, top_k=5)

    try:
        flag_reranker.stop_self_pool()
    except Exception as e:
        print(f"[Warning] stop_self_pool failed: {e}")

    del flag_reranker
