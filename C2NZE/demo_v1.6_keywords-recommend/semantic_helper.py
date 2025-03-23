import json
import re
from typing import List, Dict, Tuple

import gradio as gr
from gradio_checkboxgroupmarkdown import CheckboxGroupMarkdown


import redis
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from translation_logic import call_local_qwen_with_instruction
from translation_logic import write_log


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
    '''
    prompt = (
        "你是一位精通中文古典诗歌与英语文化的翻译顾问。\n"
        "请根据下列诗歌内容提取5~8个用于指导英文翻译的关键词或意象短语，\n"
        "请不要逐句照搬原诗句，而是提取其中可用于翻译时的意象、文化概念或核心主题词。\n"
        "并返回 JSON 格式，关键词应具有翻译价值与文化象征性。\n"
        "JSON 示例格式如下：\n"
        "{\n  \"keywords\": [\n    \"moonlight\",\n    \"bed\",\n    \"homesickness\"\n  ]\n}\n\n"
        f"诗歌原文：{poem_text}"
    ).strip()
    '''
    prompt = (
        "你是一位精通中文古典诗歌与英语文化的翻译顾问。\n"
        "请根据下列诗歌内容及长度提取适量的关键词，用于指导英文翻译。\n"
        "要求:\n"
        "1.不要逐句照搬原诗句，结合诗歌原意和意境，提取其中可用于翻译的核心主题词，包括动词、名词、形容词、副词等，以及意象和文化概念。\n"
        "2.在不破坏诗歌本意和意境的前提下，提取的主题词是按中文诗歌韵律和语义停顿划分出的最小语素和词语，特殊情况下为不破坏诗歌整体含义也可以是中文短语。\n"
        "3.主题词包括所有意象，并最终以准确简明的英文单词展示。\n"
        "4.并返回 JSON 格式，关键词应具有翻译价值与文化象征性。\n"
        "JSON 示例格式如下：\n"
        "{\n  \"keywords\": [\n    \"word1\",\n    \"word2\",\n    \"word3\"\n  ]\n}\n\n"
        f"诗歌原文：{poem_text}"
    ).strip()
    return prompt

def extract_keywords_with_llm(prompt_text: str, max_new_tokens: int = 128) -> str:
    response = call_local_qwen_with_instruction(prompt_text, max_new_tokens=max_new_tokens)
    
    # 记录完整响应到日志
    try:
        # from translation_logic import write_log
        write_log("[关键词提示完整返回]\n" + response)
    except Exception:
        print("[日志记录失败] 未能调用 write_log")

    # 提取 markdown 中的 JSON 内容
    try:
        match = re.search(r"```json\s*({[\s\S]*?})\s*", response)
        # match = re.search(r"```json\\s*(\\{[\\s\\S]*?\\})\\s*```", response)
        if match:
            return match.group(1)
            write_log(f"[JSON解析成功] {match.group(1)}")
        else:
            # return "{}"
            return response
    except Exception as e:
        print(f"[JSON解析失败] {e}")
        write_log(f"[JSON解析失败] {e}")
        return "{}"
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


def query_related_terms_from_redis(json_text: str, top_k: int = 5, model_id: int = 1) -> List[Dict]:
    keywords = display_keyword_options(json_text)  # 从 JSON 提取关键词
    all_data = []

    for kw in keywords:
        related_items = []
        topk = search_topk_similar_batch([kw], top_k=top_k, model_id=model_id)

        for i, (redis_key, score) in enumerate(topk):
            word_base = redis_key.split("-")[0]
            meaning, example_text = "(无解释)", ["(无例句)"]

            dict_data_raw = redis_dict.get(redis_key)
            if dict_data_raw:
                try:
                    dict_data = json.loads(dict_data_raw)
                    meaning = dict_data.get("meaning", meaning)
                    examples = dict_data.get("examples", [])
                    if isinstance(examples, list):
                        example_text = examples[:2]
                except Exception:
                    pass

            related_items.append({
                "word": word_base,
                "explanation": meaning,
                "examples": example_text,
                "redis_key": redis_key,
                "score": score
            })

        all_data.append({
            "keyword": kw,
            "topk": related_items
        })

    return all_data  # ⚠️ 仅返回纯 JSON，不要包含组件

def render_checkbox_groups_by_keyword(all_data: list):
    """
    输入：结构化关键词数据
    输出：List[gr.update]，用于更新预定义的 Markdown CheckboxGroup
    """
    updates = []

    for i, item in enumerate(all_data):
        keyword = item.get("keyword", f"关键词{i+1}")
        topk = item.get("topk", [])
        choices = []

        for j, entry in enumerate(topk):
            choices.append({
                "id": f"{i}_{j}",
                "title": entry["word"],
                "content": f"### Explanation\n{entry['explanation']}\n\n### Example\n" + "\n".join(entry["examples"]),
                "selected": False
            })

        updates.append(gr.update(choices=choices, visible=True, label=f"{keyword} 的相关词推荐"))

    # 若不足 10 个，隐藏多余组件
    while len(updates) < 10:
        updates.append(gr.update(choices=[], visible=False, label=""))

    return updates


def collect_grouped_markdown_selection(*args) -> str:
    """
    接收 N 个 Markdown checkbox 的 .value（List[str]），以及 all_data（List[Dict]）

    args = [cb1_value, cb2_value, ..., all_data]
    """
    *group_values, all_data = args
    result = {}

    for i, selected_ids in enumerate(group_values):
        if i >= len(all_data):
            continue
        keyword = all_data[i].get("keyword", f"关键词{i+1}")
        topk_items = all_data[i].get("topk", [])

        id_to_entry = {
            f"{i}_{j}": entry for j, entry in enumerate(topk_items)
        }

        result[keyword] = []
        for id_ in selected_ids:
            entry = id_to_entry.get(id_)
            if entry:
                result[keyword].append({
                    "word": entry["word"],
                    "explanation": entry["explanation"],
                    "examples": entry["examples"]
                })

    return json.dumps(result, ensure_ascii=False, indent=2)






def inject_keywords_into_prompt(prompt: str, selected_json: str) -> str:
    try:
        selected_items = json.loads(selected_json)
    except Exception as e:
        return prompt + "\n\n⚠️ [关键词注入失败] JSON 解析错误"

    context_parts = []
    for keyword, entries in selected_items.items():
        terms = [f"{entry['word']} ({entry['explanation']})" for entry in entries]
        context_parts.append(f"{keyword}: " + ", ".join(terms))

    context = "以下是与翻译相关的新西兰文化关键词及释义：\n" + "\n".join(context_parts)
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
