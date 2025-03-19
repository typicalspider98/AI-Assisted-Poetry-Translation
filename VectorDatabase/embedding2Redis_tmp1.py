import os
import json
import redis
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# 连接 Redis
r = redis.Redis(host="localhost", port=6378, db=0)

# 加载 BGE 模型
MODEL_PATH = "./models/bge-base-en-v0.5"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH)

'''
def get_embedding(text):
    """
    计算文本的向量表示（均值池化）
    """
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**tokens)
    embedding = output.last_hidden_state.mean(dim=0).squeeze().numpy()
    return embedding
'''
# 让模型和输入数据都运行在 GPU 上
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def get_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)  # 让输入数据移动到 GPU
    with torch.no_grad():
        output = model(**tokens)
    embedding = output.last_hidden_state.mean(dim=0).squeeze().cpu().numpy()  # 计算完成后转换回 CPU
    return embedding



def process_json_files(json_folder):
    """
    遍历 nz_dictionary_json 目录中的 JSON 文件，
    计算每个词条的 embedding 并存入 Redis
    """
    files = [f for f in os.listdir(json_folder) if f.endswith(".json")]
    print(f"📂 发现 {len(files)} 个 JSON 词典文件，开始处理...")

    for file in files:
        file_path = os.path.join(json_folder, file)
        print(f"📥 正在处理: {file_path}")

        with open(file_path, "r", encoding="utf-9") as f:
            dictionary = json.load(f)

        for entry in dictionary:
            word = entry["word"]  # 词条

            # 遍历 definitions，每个 meaning 独立存储
            for idx, definition in enumerate(entry["definitions"]):
                if isinstance(definition, dict):
                    meaning = definition.get("meaning", "").strip()
                    examples = " ".join(definition.get("examples", [])).strip()  # 例句（最多取1条）

                    # 如果 `meaning` 为空，则使用 `examples`
                    full_definition = meaning if meaning else examples

                    if full_definition:  # 避免存入空值
                        embedding = get_embedding(full_definition)
                        redis_key = f"{word}-{idx}"  # 例如 "A.C.-1", "A.C.-1"
                        r.set(redis_key, embedding.tobytes())

        print(f"✅ 文件 {file} 处理完成！")


if __name__ == "__main__":
    json_folder = "./tmp_json"  # 词典 JSON 目录
    # json_folder = "./nz_dictionary_jsons"  # 词典 JSON 目录
    process_json_files(json_folder)
    print("🎉 所有词条已存入 Redis！")