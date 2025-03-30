# extract_keywords_batch_folder.py

import os
import json
import re
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime
from transformers import BitsAndBytesConfig

# ========= 本地模型路径 =========
MODEL_PATH = "../C2NZE/models/DeepSeek-R1-Distill-Qwen-14B"

# ========= 日志配置 =========
if not os.path.exists("logs"):
    os.makedirs("logs")
log_path = os.path.join("logs", f"extract_keywords_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

def log(message: str):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message.strip() + "\n")
    print(message)

# ========= 加载模型 =========
log(f"🚀 加载模型：{MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
)
log("✅ 模型加载完成")

# ========= 核心函数 =========
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
        f"诗歌原文：{poem_text}"
    )

def call_model(prompt: str, max_new_tokens: int = 2048) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id
    )
    input_len = inputs["input_ids"].shape[-1]
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

def extract_json_keywords(model_output: str) -> List[str]:
    try:
        match = re.search(r"```json\s*({[\s\S]*?})\s*", model_output)
        if match:
            return json.loads(match.group(1)).get("keywords", [])
        else:
            return []
    except Exception as e:
        log(f"❌ JSON 解析失败: {e}")
        return []

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
        model_output = call_model(prompt)
        log(f"🔍 模型输出：\n{model_output}")
        keywords = extract_json_keywords(model_output)

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

def process_poem_folder(input_folder: str, output_folder: str):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            log(f"===============================\n📂 正在处理文件：{filename}")
            process_single_poem_file(input_path, output_path)

# ========= 启动入口 =========
if __name__ == "__main__":
    process_poem_folder("converted_jsons", "keywords_jsons")
