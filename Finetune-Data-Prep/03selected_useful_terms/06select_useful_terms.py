# select_useful_terms_with_api.py

import os
import json
from datetime import datetime
from openai import OpenAI

# ========= API 配置 =========
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "sk-e5484044e6314d95b63af7f93a00ea7e"),
    base_url="https://api.deepseek.com"
)

# ========= 日志配置 =========
if not os.path.exists("logs"):
    os.makedirs("logs")
log_path = os.path.join("logs", f"select_terms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

def log(message: str):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message.strip() + "\n")
    print(message)

# ========= 构建提示词 =========
def build_selection_prompt(poem: dict) -> str:
    prompt = (
        "你是一位专业的新西兰本地化英语翻译顾问。\n"
        "请根据以下古诗内容、关键词及其关联同义词，选择其中对本诗英文翻译最有帮助的相关词。\n\n"
        "要求：\n"
        "1. 你的任务是：从每组 TopK 推荐词中选择 0-2 个最符合本诗意境的词；\n"
        "2. 如果该组中无合适者，可不选择；\n"
        "3. 输出 JSON，结构如下：\n"
        "```json\n{\n  \"title\": \"...\",\n  \"useful_keywords\": [\"keyword1\", \"keyword2\"],\n  \"selected_terms\": {\n    \"keyword1\": [\"selected_word1\"],\n    \"keyword2\": []\n  }\n}\n```\n"
        f"\n古诗内容：{poem['content']}\n\n"
        f"关键词列表：{json.dumps(poem['keywords'], ensure_ascii=False)}\n\n"
        f"每个关键词对应的推荐词如下：\n{json.dumps(poem['related_terms'], ensure_ascii=False, indent=2)}"
    )
    return prompt

# ========= 调用 API =========
def call_selection_api(prompt: str, max_tokens: int = 2048) -> dict:
    messages = [
        {"role": "system", "content": "你是一位新西兰本地化翻译顾问，请以 JSON 格式回答。"},
        {"role": "user", "content": prompt}
    ]
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            stream=False
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        log(f"❌ API 调用或解析失败: {e}")
        return {}

# ========= 主处理逻辑 =========
def process_enriched_poem_file(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        poems = json.load(f)

    selected_results = []
    for i, poem in enumerate(poems):
        log(f"\n📜 正在处理第{i+1}首诗：《{poem.get('title', '')}》")
        prompt = build_selection_prompt(poem)
        result = call_selection_api(prompt)

        selected_terms = result.get("selected_terms", {})
        # 过滤掉空选择
        selected_terms = {k: v for k, v in selected_terms.items() if v}
        useful_keywords = list(selected_terms.keys())

        record = {
            "title": poem.get("title", f"poem_{i+1}"),
            "author": poem.get("author", ""),
            "dynasty": poem.get("dynasty", ""),
            "content": poem.get("content", ""),
            "keywords": poem.get("keywords", []),
            "related_terms": poem.get("related_terms", []),
            "selected_terms": selected_terms,
            "useful_keywords": useful_keywords
        }
        selected_results.append(record)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selected_results, f, ensure_ascii=False, indent=2)
    log(f"✅ 所有结果已保存至：{output_path}")

# ========= 示例调用 =========
if __name__ == "__main__":
    process_enriched_poem_file(
        "keywords_enriched_jsons_DeepSeek_output/唐_part1_enriched.json",
        "final_selected_terms_output/唐_part1_selected.json"
    )
