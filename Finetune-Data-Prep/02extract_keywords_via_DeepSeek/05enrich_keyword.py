import os
import json
from typing import List, Dict

import redis
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# ========= Redis 向量与释义数据库 =========
redis_vec = redis.Redis(host="localhost", port=6379, db=2)
redis_dict = redis.Redis(host="localhost", port=6379, db=0)

# ========= 嵌入模型配置 =========
EMBEDDING_MODELS = {
    2: "/workspace/AI-Assisted-Poetry-Translation/VectorDatabase/models/bge-m3",
}
_loaded_models = {}

def get_embedding(text: str, model_id: int = 2):
    if model_id not in _loaded_models:
        path = EMBEDDING_MODELS[model_id]
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModel.from_pretrained(path).to("cuda")
        _loaded_models[model_id] = (tokenizer, model)

    tokenizer, model = _loaded_models[model_id]
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    tokens = {k: v.to(model.device) for k, v in tokens.items()}
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def search_topk_similar_batch(queries: List[str], top_k: int = 6, model_id: int = 2):
    merged_query = ", ".join(queries)
    query_vector = get_embedding(merged_query, model_id)
    query_norm = np.linalg.norm(query_vector)
    results = []
    for key in redis_vec.keys("*"):
        stored_vector = np.frombuffer(redis_vec.get(key), dtype=np.float32)
        stored_norm = np.linalg.norm(stored_vector)
        if stored_norm == 0 or query_norm == 0:
            score = 0.0
        else:
            score = np.dot(query_vector, stored_vector) / (query_norm * stored_norm)
        results.append((key.decode("utf-8"), score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

def extract_base_word(redis_key: str) -> str:
    parts = redis_key.rsplit("-", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else redis_key

def query_related_terms_from_redis(keywords: List[str], top_k: int = 6, model_id: int = 2) -> List[Dict]:
    results = []
    for kw in keywords:
        related = []
        topk = search_topk_similar_batch([kw], top_k=top_k, model_id=model_id)
        for redis_key, score in topk:
            base_word = extract_base_word(redis_key)
            meaning = "(无解释)"
            examples = ["(无例句)"]
            raw = redis_dict.get(redis_key)
            if raw:
                try:
                    data = json.loads(raw)
                    meaning = data.get("meaning", meaning)
                    ex = data.get("examples", [])
                    if isinstance(ex, list):
                        examples = ex[:2]
                except Exception:
                    pass
            related.append({
                "word": base_word,
                "explanation": meaning,
                "examples": examples,
                "redis_key": redis_key,
                "score": float(score)
            })
        results.append({"keyword": kw, "topk": related})
    return results

def convert_keywords_to_list(keywords: List[str]) -> List[str]:
    return [k.split(":")[0].strip() for k in keywords if ":" in k]

def enrich_poems_with_related_terms(input_path: str, output_path: str, top_k: int = 6):
    with open(input_path, "r", encoding="utf-8") as f:
        poems = json.load(f)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    enriched = []
    for poem in poems:
        keywords = convert_keywords_to_list(poem.get("keywords", []))
        if not keywords:
            continue
        related = query_related_terms_from_redis(keywords, top_k=top_k, model_id=2)
        poem["related_terms"] = related
        enriched.append(poem)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2, default=lambda o: float(o) if isinstance(o, np.float32) else str(o))

    print(f"✅ 已写入 {len(enriched)} 首诗到 {output_path}")

# 示例调用
enrich_poems_with_related_terms(
    "keywords_jsons_DeepSeek_output/唐_part1.json",
    "keywords_enriched_jsons_DeepSeek_output/唐_part1_enriched.json"
)