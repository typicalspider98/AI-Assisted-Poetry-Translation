# 改进后的 semantic_helper.py（采用延迟加载模型，解决多进程冲突问题）

import json
import re
from typing import List, Dict, Tuple

import gradio as gr
from gradio_checkboxgroupmarkdown import CheckboxGroupMarkdown

import redis
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from FlagEmbedding import FlagReranker
from transformers import AutoModelForSequenceClassification

from translation_logic import call_local_qwen_with_instruction
from translation_logic import write_log

# === Redis 配置 ===
redis_vec = redis.Redis(host="localhost", port=6379, db=2)
redis_dict = redis.Redis(host="localhost", port=6379, db=0)

# 支持多个嵌入模型路径
EMBEDDING_MODELS = {
    2: "../../VectorDatabase/models/bge-m3",
}
_loaded_models = {}

# === 延迟加载 reranker 模型 ===
_flag_reranker = None
_gte_tokenizer = None
_gte_model = None

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
def build_keyword_prompt_EN(poem_text: str, max_new_tokens: int = 128) -> str:
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
        f"You are a translation advisor well-versed in classical Chinese poetry and English cultural expression.\n"
        f"Based on the following poem and its length, extract a suitable set of thematic keywords to guide the English translation process.\n"
        f"Requirements:\n"
        f"1. Do not simply extract phrases directly from each line of the original poem.\n"
        f"2. Instead, analyze the overall meaning and imagery to identify core thematic words that can assist in capturing the essence of the poem in translation. "
        f"These may include verbs, nouns, adjectives, adverbs, as well as symbolic imagery and culturally relevant concepts.\n"
        f"3. Extract keywords according to natural pauses and semantic breaks in the poem's structure, following the rhythm and meaning of the original Chinese.\n"
        f"4. In special cases, short Chinese phrases may be accepted as a single unit if breaking them apart would compromise the poem's overall meaning or imagery.\n"
        f"5. Include all key symbols and imagery from the poem, and present the extracted thematic keywords as accurate, concise English words.\n"
        f"6. Return the results in JSON format. The selected keywords should carry both translational value and cultural symbolism.\n"
        f"Expected JSON format:\n"
        "{\n  \"keywords\": [\n    \"word1: English simple explanation\",\n    \"word2: English simple explanation\",\n    \"word3: English simple explanation\"\n  ]\n}\n\n"
        f"Original Poem:{poem_text}"
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


# === 向量生成 ===

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
    embedding = output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().astype(np.float32)
    return embedding

# === TopK检索与重排序 ===
# === 延迟加载 reranker 辅助函数 ===

def custom_sigmoid(x, temperature=1.5):
    return 1 / (1 + np.exp(-temperature * x))

def get_flag_reranker():
    global _flag_reranker
    if _flag_reranker is None:
        print("[加载] FlagEmbedding reranker...")
        _flag_reranker = FlagReranker("../models/bge-reranker-v2-m3", use_fp16=True)
    return _flag_reranker

def get_gte_reranker():
    global _gte_tokenizer, _gte_model
    if _gte_tokenizer is None or _gte_model is None:
        print("[加载] GTE reranker...")
        _gte_tokenizer = AutoTokenizer.from_pretrained("../models/gte-multilingual-reranker-base")
        _gte_model = AutoModelForSequenceClassification.from_pretrained(
            "../models/gte-multilingual-reranker-base", trust_remote_code=True, torch_dtype=torch.float16
        ).eval().to("cuda")
    return _gte_tokenizer, _gte_model
def rerank_with_flag(pairs: List[List[str]]) -> List[float]:
    model = get_flag_reranker()
    scores = model.compute_score(pairs, normalize=False)
    return [custom_sigmoid(x, temperature=2.0) for x in scores]

def rerank_with_gte(pairs: List[List[str]]) -> List[float]:
    tokenizer, model = get_gte_reranker()
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to("cuda")
        logits = model(**inputs).logits.view(-1).float()
        return [custom_sigmoid(x.item(), temperature=2.0) for x in logits]

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



def search_topk_with_reranker(queries: List[str], top_k: int = 6, model_id: int = 2) -> List[Dict]:
    """
    左右位置颠倒版：左边是kw（关键词），右边是Redis查到的词条+释义
    """

    results = []

    for raw_kw in queries:
        # === 提取关键词（只要冒号前的）
        if ":" in raw_kw:
            kw = raw_kw.split(":")[0].strip()
        else:
            kw = raw_kw.strip()

        # [1] 用kw去Redis查TopK
        topk = search_topk_similar_batch([kw], top_k=top_k*2, model_id=model_id)

        pairs = []
        redis_keys = []
        cos_sims = []
        passages = []

        for key, cos_sim in topk:
            word_base = key.rsplit("-", 1)[0]
            meaning = "(无解释)"

            if redis_dict.exists(key):
                try:
                    entry = json.loads(redis_dict.get(key))
                    meaning = entry.get("meaning", meaning)
                except:
                    pass

            message = f"{word_base}: {meaning}" if meaning != "(无解释)" else word_base

            # ✅ 改这里：左边是 kw，右边是 message
            pairs.append([kw, message])

            passages.append(message)
            redis_keys.append(key)
            cos_sims.append(cos_sim)

        # [2] reranker打分
        print("\n--- 当前pairs内容 ---")
        for p in pairs:
            print(f"左边(query=kw): {p[0]} || 右边(message): {p[1]}")
        print("--- 结束 ---\n")

        scores_flag = rerank_with_flag(pairs)
        scores_gte = rerank_with_gte(pairs)

        # [3] 保存打分
        for i in range(len(pairs)):
            results.append({
                "keyword": kw,  # 查询词
                "redis_key": redis_keys[i],
                "cosine_similarity": cos_sims[i],
                "flag_score": scores_flag[i],
                "gte_score": scores_gte[i],
                "final_score": max(scores_flag[i], scores_gte[i])
            })

    # [4] 排序
    results.sort(key=lambda x: x["final_score"], reverse=True)

    return results[:top_k]



def search_topk_with_reranker_demo0(queries: List[str], top_k: int = 6, model_id: int = 2) -> List[Dict]:
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
            similarity = float(np.dot(query_vector, stored_vector)) / (query_norm * stored_norm)
        similarities.append((word.decode("utf-8"), similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    topk = similarities[:top_k * 2]

    pairs = []
    for key, _ in topk:
        word = key.rsplit("-", 1)[0]
        '''
        meaning = "(无解释)"
        if redis_dict.exists(key):
            try:
                entry = json.loads(redis_dict.get(key))
                meaning = entry.get("meaning", meaning)
            except:
                pass
        passage = f"{word}: {meaning}"
        '''
        passage = word
        pairs.append([merged_query, passage])

    scores_flag = rerank_with_flag(pairs)
    scores_gte = rerank_with_gte(pairs)

    results = []
    for i, (key, cos_sim) in enumerate(topk):
        results.append({
            "redis_key": key,
            "cosine_similarity": cos_sim,
            "flag_score": scores_flag[i],
            "gte_score": scores_gte[i],
            "final_score": max(scores_flag[i], scores_gte[i])
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results[:top_k]

# === 查询相关词并组织结果 ===

def query_related_terms_from_redis(json_text: str, top_k: int = 6, model_id: int = 2) -> List[Dict]:
    if not json_text.strip():
        return []
    try:
        keywords = json.loads(json_text).get("keywords", [])
    except Exception as e:
        print(f"[解析关键词JSON失败] {e}")
        return []

    all_data = []
    for kw in keywords:
        related_items = []
        topk = search_topk_with_reranker([kw], top_k=top_k, model_id=model_id)
        for item in topk:
            redis_key = item["redis_key"]
            word_base = redis_key.rsplit("-", 1)[0]
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
                "cosine_similarity": item["cosine_similarity"],
                "flag_score": item["flag_score"],
                "gte_score": item["gte_score"],   
                "final_score": item["final_score"]
            })

        all_data.append({
            "keyword": kw,
            "topk": related_items
        })

    return all_data

# === 渲染关键词相关词为Checkbox展示 ===
def weather_icon(score: float) -> str:
    if score >= 0.70:
        return "☀️ (Excellent)"
    elif score >= 0.60:
        return "🌤️ (Good)"
    elif score >= 0.50:
        return "☁️ (Moderate)"
    elif score >= 0.40:
        return "🌧️ (Weak)"
    else:
        return "🌩️ (Poor)"
def render_checkbox_groups_by_keyword(all_data: list):
    updates = []
    for i, item in enumerate(all_data):
        keyword = item.get("keyword", f"关键词{i+1}")
        topk = item.get("topk", [])

        choices = []
        for j, entry in enumerate(topk):
            word = entry["word"]
            explanation = entry["explanation"]
            examples = entry["examples"]

            cosine = entry.get("cosine_similarity", 0.0)
            flag = entry.get("flag_score", 0.0)
            gte = entry.get("gte_score", 0.0)
            final = entry.get("final_score", 0.0)
            
            # 把四个分数一起展示
            # score = entry.get("score", 0.0)
            # score_display = f"⭐️ 相似度：{score:.3f}" if score > 0.75 else f"相似度：{score:.3f}"
            '''
            score_display = (
                f"余弦相似度 (cos): {cosine:.3f}, "
                f"Flag得分: {flag:.3f}, "
                f"GTE得分: {gte:.3f}, "
                f"最终得分: {final:.3f}"
            )
            '''
            score_display = (
                "|  cos-Sim | Flag-Score | GTE-Score | **Final Score** |\n"
                "| :---: | :---: | :---: | :---: |\n"
                # f"| {cosine:.3f} | {flag:.3f} | {gte:.3f} | {'⭐️ ' if final > 0.6 else ''}{final:.3f} |"
                f"| {cosine:.3f} | {flag:.3f} | {gte:.3f} | **{final:.3f}** |"
            )

    
            choices.append({
                "id": f"{i}_{j}",
                "title": explanation,
                "content": f"### {word} {weather_icon(final)}\n{score_display}\n\n" + "\n".join(f"- {ex}" for ex in examples),
                "selected": False
            })

        half = (len(choices) + 1) // 2
        left_choices = choices[:half]
        right_choices = choices[half:]

        updates.append([
            gr.update(choices=left_choices, visible=True, label=f"{keyword}"),
            gr.update(choices=right_choices, visible=True, label=f"{keyword}")
        ])

    while len(updates) < 50:
        updates.append([
            gr.update(choices=[], visible=False, label=""),
            gr.update(choices=[], visible=False, label="")
        ])

    return [item for pair in updates for item in pair]

# （其他collect_grouped_markdown_selection, update_accordion_labels, inject_keywords_into_prompt保持不变）


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