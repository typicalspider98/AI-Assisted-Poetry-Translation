# build_finetune_dataset.py

import os
import json
from typing import List

# ========= 构造训练样本 =========
def format_keywords(keywords: List[str]) -> List[str]:
    # 确保关键词格式是 word: explanation 的形式
    return [kw.strip() for kw in keywords if ":" in kw]

def build_finetune_samples(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        poems = json.load(f)

    samples = []
    for poem in poems:
        content = poem.get("content", "").strip()
        keywords = format_keywords(poem.get("keywords", []))
        useful_keywords = poem.get("useful_keywords", [])

        # 从关键词中筛选出只在 useful_keywords 中的
        filtered = [k for k in keywords if k.split(":")[0].strip() in useful_keywords]
        if not filtered:
            continue

        sample = {
            "instruction": "你是一位精通中文古典诗歌与英语文化的翻译顾问。\n"
                            "请根据下列诗歌内容及长度提取适量的关键词，用于指导英文翻译。\n"
                            "要求:\n"
                            "1.不要逐句照搬原诗句，结合诗歌原意和意境，提取其中可用于翻译的核心主题词，包括动词、名词、形容词、副词等，以及意象和文化概念。\n"
                            "2.在不破坏诗歌本意和意境的前提下，提取的主题词是按中文诗歌韵律和语义停顿划分出的最小语素和词语，特殊情况下为不破坏诗歌整体含义也可以是中文短语。\n"
                            "3.主题词包括所有意象，并最终以准确简明的英文单词展示。\n"
                            "4.并返回 JSON 格式，关键词应具有翻译价值与文化象征性。\n\n"
                            "JSON 示例格式如下：\n"
                            "{\n  \"keywords\": [\n    \"word1: English simple explanation\",\n    \"word2: English simple explanation\"\n  ]\n}\n",
            "input": content,
            "output": json.dumps({"keywords": filtered}, ensure_ascii=False)
        }
        samples.append(sample)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"✅ 已构建 {len(samples)} 条训练样本，保存至 {output_path}")

# ========= 示例调用 =========
if __name__ == "__main__":
    build_finetune_samples(
        "final_selected_terms_output/唐_part1_selected.json",
        "finetune_dataset/唐_part1_lora.jsonl"
    )
