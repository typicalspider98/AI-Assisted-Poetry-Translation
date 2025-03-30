import os
import json
import re
from typing import List
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
log_path = os.path.join("logs", f"extract_keywords_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

def log(message: str):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message.strip() + "\n")
    print(message)

# ========= 构建提示词 =========
def build_keyword_prompt(poem_text: str) -> str:
    return (
        "你是一位精通中文古典诗歌与英语文化的翻译顾问。\n"
        "请根据下列诗歌内容及长度提取适量的关键词<word>，用于指导英文翻译。\n"
        "要求:\n"
        "1.不要逐句照搬原诗句，结合诗歌原意和意境，提取其中可用于翻译的核心主题词，例如：动词、名词、形容词、副词等，以及意象和文化概念。\n"
        "2.在不破坏诗歌本意和意境的前提下，提取的主题词<word>，按中文诗歌韵律和语义停顿划分出的最小语素和词语。\n"
        "3.主题词<word>包括所有意象，为简明的英文单词。\n"
        "4.返回 JSON 格式，关键词应具有翻译价值与文化象征性。\n"
        "JSON 示例格式如下：\n"
        "```json\n{\n  \"keywords\": [\n    \"<word>: Simple explanation in English>\",\n    \"<word>: <Simple explanation in English>\",\n"
        "    \"<word>: <Simple explanation in English>\"\n  ]\n}\n```\n\n"
        f"\n诗歌原文：{poem_text}"
    )

# ========= 远程调用 API =========
def call_deepseek_api(prompt: str, max_tokens: int = 2048) -> List[str]:
    messages = [
        {
            "role": "system",
            "content": "你是一位精通中文古典诗歌与英语文化的翻译顾问，请以合法 JSON 格式回答。"
        },
        {
            "role": "user",
            "content": prompt
        }
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
        return result.get("keywords", [])
    except Exception as e:
        log(f"❌ API 调用或 JSON 解析失败: {e}")
        return []

# ========= 处理单个文件 =========
def process_single_poem_file(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        poems = json.load(f)

    all_results = []
    for i, poem in enumerate(poems):
        content = poem.get("content", "").strip()
        if not content:
            log(f"⚠️ 第{i+1}首诗内容为空，跳过")
            continue

        log(f"\n📜 处理第{i+1}首：《{poem.get('title', '')}》")
        prompt = build_keyword_prompt(content)
        keywords = call_deepseek_api(prompt)

        if keywords:
            poem_result = {
                "title": poem.get("title", f"poem_{i+1}"),
                "author": poem.get("author", ""),
                "dynasty": poem.get("dynasty", ""),
                "content": content,
                "keywords": keywords
            }
            all_results.append(poem_result)
            log(f"✅ 提取关键词：{keywords}")
        else:
            log(f"⚠️ 未提取出关键词")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    log(f"📁 已保存至：{output_path}")

# ========= 批量处理 =========
def process_poem_folder(input_folder: str, output_folder: str):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            log(f"===============================\n📂 正在处理文件：{filename}")
            process_single_poem_file(input_path, output_path)

# ========= 入口 =========
if __name__ == "__main__":
    process_poem_folder("jsons_extracting", "keywords_jsons_DeepSeek_output")
