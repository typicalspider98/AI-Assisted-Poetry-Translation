import json
import re
from typing import List, Dict

import redis
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from translation_logic import call_local_qwen_with_instruction

# === Redis 数据库配置 ===
redis_vec = redis.Redis(host="localhost", port=6379, db=2)  # 向量数据库
redis_dict = redis.Redis(host="localhost", port=6379, db=0)  # 词典释义数据库

# 支持多个嵌入模型路径
EMBEDDING_MODELS = {
    1: "/workspace/Project-Code/AI-Assisted-Poetry-Translation/VectorDatabase/models/bge-base-en-v1.5",
    2: "/workspace/Project-Code/AI-Assisted-Poetry-Translation/VectorDatabase/models/bge-m3",
}

# 根据 model_id 加载对应 embedding 模型（缓存）
_loaded_models = {}

def get_embedding(text: str, model_id: int = 1):
    if model_id not in _loaded_models:
        path = EMBEDDING_MODELS.get(model_id)
        if not path:
            raise ValueError(f"❌ 未知的嵌入模型编号: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModel.from_pretrained(path).to("cuda")
        _loaded_models[model_id] = (tokenizer, model)

    tokenizer, model = _loaded_models[model_id]
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    tokens = {k: v.to(model.device) for k, v in tokens.items()}

    with torch.no_grad():
        output = model(**tokens)
    embedding = output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding


def search_topk_similar_batch(queries: List[str], top_k: int = 5, model_id: int = 1):
    merged_query = ", ".join(queries)
    query_vector = get_embedding(merged_query, model_id)
    words = redis_vec.keys("*")
    similarities = []
    for word in words:
        stored_vector = np.frombuffer(redis_vec.get(word), dtype=np.float32)
        similarity = np.dot(query_vector, stored_vector)
        similarities.append((word.decode("utf-8"), similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


# === 功能模块 ===

def build_keyword_prompt(poem_text: str, max_new_tokens: int = 128) -> str:
    prompt = (
        "你是一位精通中文古典诗歌与英语文化的翻译顾问。\n"
        "请根据下列诗歌内容提取5~8个用于指导英文翻译的关键词或意象短语，\n"
        "请不要逐句照搬原诗句，而是提取其中可用于翻译时的意象、文化概念或核心主题词。\n"
        "并返回 JSON 格式，关键词应具有翻译价值与文化象征性。\n"
        "JSON 示例格式如下：\n"
        "{\n  \"keywords\": [\n    \"moonlight\",\n    \"bed\",\n    \"homesickness\"\n  ]\n}\n\n"
        f"诗歌原文：{poem_text}"
    ).strip()
    return prompt

def extract_keywords_with_llm(prompt_text: str, max_new_tokens: int = 128) -> str:
    response = call_local_qwen_with_instruction(prompt_text, max_new_tokens=max_new_tokens)
    return response


def extract_keywords_regex(poem_text: str) -> str:
    keywords = list(set(re.findall(r'[\u4e00-\u9fff]{2,4}', poem_text)))
    return json.dumps({"keywords": keywords}, ensure_ascii=False, indent=2)


def display_keyword_options(json_text: str) -> List[str]:
    try:
        data = json.loads(json_text)
        return data.get("keywords", [])
    except Exception:
        return []


def query_related_terms_from_redis(keywords: List[str], top_k: int = 5, model_id: int = 1) -> List[Dict]:
    results = []
    for kw in keywords:
        topk = search_topk_similar_batch([kw], top_k=top_k, model_id=model_id)
        for i, (redis_key, score) in enumerate(topk):
            word_base = redis_key.split("-")[0]

            # 从 db=0 查询解释信息
            dict_data_raw = redis_dict.get(redis_key)
            meaning, example_text = "(无解释)", "(无例句)"
            if dict_data_raw:
                try:
                    dict_data = json.loads(dict_data_raw)
                    meaning = dict_data.get("meaning", meaning)
                    examples = dict_data.get("examples", [])
                    if isinstance(examples, list):
                        example_text = "\n".join(examples[:2])
                except Exception:
                    pass

            content = f"### Explanation\n{meaning}\n\n### Example\n{example_text}"

            results.append({
                "id": f"{kw}_{i}",
                "title": word_base,
                "content": content,
                "selected": False
            })
    return results


def inject_keywords_into_prompt(prompt: str, selected_items: List[Dict]) -> str:
    selected_titles = [item["title"] for item in selected_items]
    context = "以下是与翻译相关的新西兰文化关键词：\n" + ", ".join(selected_titles) + "\n请结合这些信息进行翻译。"
    return prompt.strip() + "\n\n" + context


if __name__ == "__main__":
    poem = "床前明月光，疑是地上霜。举头望明月，低头思故乡。"
    print("\n[1] 正在提取关键词...")
    keywords_json = extract_keywords_with_llm(poem)
    print(keywords_json)

    keywords = display_keyword_options(keywords_json)
    print("\n[2] 用户选择关键词:", keywords)

    print("\n[3] 查询相关词...")
    related_terms = query_related_terms_from_redis(keywords, model_id=1)
    for item in related_terms:
        print(f"- {item['title']}\n{item['content']}\n")

    print("\n[4] 构造增强提示:")
    enriched_prompt = inject_keywords_into_prompt("请翻译这首诗：\n" + poem, related_terms[:3])
    print(enriched_prompt)
