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
    # 1: "/workspace/Project-Code/AI-Assisted-Poetry-Translation/VectorDatabase/models/bge-base-en-v1.5",
    2: "/workspace/AI-Assisted-Poetry-Translation/VectorDatabase/models/bge-m3",
}

# 根据 model_id 加载对应 embedding 模型（缓存）
_loaded_models = {}

def get_embedding(text: str, model_id: int = 2):
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
    # embedding = output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    embedding = output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().astype(np.float32)
    return embedding

'''
def search_topk_similar_batch(queries: List[str], top_k: int = 6, model_id: int = 1):
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

def search_topk_similar_batch(queries: List[str], top_k: int = 6, model_id: int = 2):
    merged_query = ", ".join(queries)
    query_vector = get_embedding(merged_query, model_id)
    query_norm = np.linalg.norm(query_vector)
    
    words = redis_vec.keys("*")
    similarities = []
    for word in words:
        stored_vector = np.frombuffer(redis_vec.get(word), dtype=np.float32)
        stored_norm = np.linalg.norm(stored_vector)
        if stored_norm == 0 or query_norm == 0:
            similarity = 0.0
        else:
            similarity = np.dot(query_vector.astype(np.float32), stored_vector.astype(np.float32)) / (query_norm * stored_norm)
            # similarity = np.dot(query_vector, stored_vector) / (query_norm * stored_norm)
        similarities.append((word.decode("utf-8"), similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]
'''
def search_topk_similar_batch(queries: List[str], top_k: int = 6, model_id: int = 2):
    merged_query = ", ".join(queries)
    query_vector = get_embedding(merged_query, model_id)
    query_vector = np.asarray(query_vector, dtype=np.float32).flatten()
    query_norm = np.linalg.norm(query_vector)

    words = redis_vec.keys("*")
    similarities = []
    for word in words:
        stored_vector = np.frombuffer(redis_vec.get(word), dtype=np.float32).flatten()
        stored_norm = np.linalg.norm(stored_vector)

        if stored_norm == 0 or query_norm == 0:
            similarity = 0.0
        else:
            similarity = float(np.dot(query_vector, stored_vector)) / (query_norm * stored_norm)

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
        "{\n  \"keywords\": [\n    \"word1: English simple explanation\",\n    \"word2: English simple explanation\",\n    \"word3: English simple explanation\"\n  ]\n}\n\n"
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

def extract_base_word(redis_key: str) -> str:
    """
    从 Redis key 中提取词条本体，剥去末尾的 -数字 部分（如 'gumboot-0' → 'gumboot'）
    注意词条本身可能包含连字符（如 'bum-cheek salute-12'）
    """
    parts = redis_key.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return redis_key  # fallback: key 不符合结构，直接返回原始 key


def query_related_terms_from_redis(json_text: str, top_k: int = 6, model_id: int = 2) -> List[Dict]:
    keywords = display_keyword_options(json_text)  # 从 JSON 提取关键词
    all_data = []

    for kw in keywords:
        related_items = []
        topk = search_topk_similar_batch([kw], top_k=top_k, model_id=model_id)

        for i, (redis_key, score) in enumerate(topk):
            # word_base = redis_key.split("-")[0]  # wrong
            # word_base = "-".join(redis_key.split("-")[:-1])  # not good enough
            word_base = extract_base_word(redis_key)
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
    输入：query_related_terms_from_redis 的返回结果（关键词+相关词）
    输出：List[List[gr.update, gr.update]] 对应每组的左右 checkbox 更新
    """
    updates = []

    for i, item in enumerate(all_data):
        keyword = item.get("keyword", f"关键词{i+1}")
        topk = item.get("topk", [])

        # Debug: 打印相关词 word 列表
        # print(f"[关键词 {i+1}] {keyword} 的相关词有：", [entry["word"] for entry in topk])
        choices = []
        for j, entry in enumerate(topk):
            word = entry["word"]
            score = entry.get("score", 0.0)
            explanation = entry["explanation"]
            examples = entry["examples"]
            # score_display = f"（相似度：{score:.3f}）"
            score_display = f"⭐️ 相似度：{score:.3f}" if score > 0.90 else f"相似度：{score:.3f}"

            '''
            choices.append({
                "id": f"{i}_{j}",
                "title": entry["explanation"],
                "content": f"### {entry['word']}\n" + "\n".join(f"- {ex}" for ex in entry["examples"]),  # "\n".join(entry["examples"]),
                "selected": False
            })
            '''
            choices.append({
                "id": f"{i}_{j}",
                "title": explanation,
                "content": f"### {word} {score_display}\n" + "\n".join(f"- {ex}" for ex in examples),
                "selected": False
            })

        # 拆成左右两列（平均分）
        half = (len(choices) + 1) // 2
        left_choices = choices[:half]
        right_choices = choices[half:]

        updates.append([
            gr.update(choices=left_choices, visible=True, label=f"{keyword}"),  # （左列）"),
            gr.update(choices=right_choices, visible=True, label=f"{keyword}"),  # （右列）")
        ])

    # 如果不足50组，补空
    while len(updates) < 50:
        updates.append([
            gr.update(choices=[], visible=False, label=""),
            gr.update(choices=[], visible=False, label="")
        ])

    # 展平成一个 list：[左0, 右0, 左1, 右1, ...]
    flat_updates = [item for pair in updates for item in pair]
    return flat_updates


def collect_grouped_markdown_selection(*args) -> str:
    """
    参数结构：
    - 前 2*N 个是 CheckboxGroupMarkdown 的 .value
    - 最后一个是 all_data（包含所有关键词组及其 TopK 释义信息）
    """
    *group_values, all_data = args
    result = {}

    for group_index in range(len(group_values) // 2):
        selected_left = group_values[group_index * 2] or []
        selected_right = group_values[group_index * 2 + 1] or []
        selected_ids = selected_left + selected_right

        if group_index >= len(all_data):
            continue

        keyword = all_data[group_index].get("keyword", f"关键词{group_index+1}")
        topk_items = all_data[group_index].get("topk", [])

        id_to_entry = {
            f"{group_index}_{j}": entry for j, entry in enumerate(topk_items)
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



def update_accordion_labels(all_related_data):
    return [
        gr.Accordion.update(label=f"关键词：{group['keyword']}", visible=True)
        for group in all_related_data
    ] + [gr.Accordion.update(visible=False)] * (50 - len(all_related_data))



'''
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
'''
def inject_keywords_into_prompt(prompt: str, selected_json: str) -> str:
    try:
        selected_items = json.loads(selected_json)
    except Exception as e:
        return prompt + "\n\n⚠️ [关键词注入失败] JSON 解析错误"

    context_parts = []
    for keyword, entries in selected_items.items():
        if not entries:
            continue  # ✅ 跳过没有选中任何相关词的关键词
        terms = [f"{entry['word']} ({entry['explanation']})" for entry in entries]
        context_parts.append(f"{keyword}: " + ", ".join(terms))

    if not context_parts:
        return prompt + "\n\n⚠️ [提示] 当前未选择任何关键词释义，未进行注入"

    context = "以下是与本首诗歌翻译相关的新西兰英语关键词及释义，可供参考：\n" + "\n".join(context_parts)
    return prompt.strip() + "\n\n" + context



if __name__ == "__main__":
    print("this is not the file you should run.\nGo find web_interface.py")
'''
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
'''