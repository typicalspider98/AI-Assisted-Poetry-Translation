# extract_keywords_batch.py
# 测试本地模型从json诗歌文本提取诗歌关键词并保存
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
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True
    )

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,  # load_in_8bit=True,  # 启用 8-bit 量化
    device_map="auto",  # device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        )
    )
'''
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
'''
log("✅ 模型加载完成")

# ========= 核心函数 =========
def build_keyword_prompt(poem_text: str) -> str:
    return (
        "你是一位精通中文古典诗歌与英语文化的翻译顾问。\n"
        "请根据下列诗歌内容及长度提取适量的关键词，用于指导英文翻译。\n"
        "要求:\n"
        "1.不要逐句照搬原诗句，结合诗歌原意和意境，提取其中可用于翻译的核心主题词，包括动词、名词、形容词、副词等，以及意象和文化概念。\n"
        "2.在不破坏诗歌本意和意境的前提下，提取的主题词是按中文诗歌韵律和语义停顿划分出的最小语素和词语，特殊情况下为不破坏诗歌整体含义也可以是中文短语。\n"
        "3.主题词包括所有意象，并最终以准确简明的英文单词展示。\n"
        "4.并返回 JSON 格式，关键词应具有翻译价值与文化象征性。\n"
        "JSON 示例格式如下：\n"
        "```json\n{\n  \"keywords\": [\n    \"word1: English simple explanation\",\n    \"word2: English simple explanation\",\n    \"word3: English simple explanation\"\n  ]\n}\n```\n\n"
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
            return []  # fallback: 不再额外处理原始输出，保持简洁清晰
    except Exception as e:
        log(f"❌ JSON 解析失败: {e}")
        return []

def load_poems(filepath: str) -> List[Dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_keywords_batch(poem_json_path: str, output_path: str):
    poems = load_poems(poem_json_path)
    all_results = []

    for i, poem in enumerate(poems):
        content = poem.get("content", "").strip()
        if not content:
            log(f"⚠️ 第{i+1}首诗内容为空，跳过")
            continue

        log(f"\n📜 处理第{i+1}首：《{poem.get('title', '')}》")
        prompt = build_keyword_prompt(content)
        model_output = call_model(prompt)
        log(f"🔍 模型输出：\n{model_output}")  # 用于调试
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
    log(f"\n📁 已保存所有结果至：{output_path}")

# ========= 启动脚本 =========
if __name__ == "__main__":
    extract_keywords_batch("poems.json", "keywords_all.json")