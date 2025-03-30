import os
import json
import redis
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# 连接 Redis
r = redis.Redis(host="localhost", port=6379, db=2)  # 建议确认使用 db=2 一致性

# 加载 BGE 模型
MODEL_PATH = "./models/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH)


def get_embedding(text):
    """
    计算文本的向量表示（均值池化）并转换为 float32 格式
    """
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**tokens)
    embedding = output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().astype(np.float32)
    return embedding


def process_json_files(json_folder):
    """
    遍历 nz_dictionary_json 目录中的 JSON 文件，
    计算每个词条的 embedding 并存入 Redis（如果 Redis 中没有才存）
    """
    files = [f for f in os.listdir(json_folder) if f.endswith(".json")]
    print(f"📂 发现 {len(files)} 个 JSON 词典文件，开始处理...")

    for file in files:
        file_path = os.path.join(json_folder, file)
        print(f"📥 正在处理: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            dictionary = json.load(f)

        for entry in dictionary:
            word = entry["word"]  # 词条

            # 遍历 definitions，每个 meaning 独立存储
            for idx, definition in enumerate(entry["definitions"]):
                if isinstance(definition, dict):
                    meaning = definition.get("meaning", "").strip()
                    examples = " ".join(definition.get("examples", [])).strip()  # 例句（最多取2条）

                    # 如果 `meaning` 为空，则使用 `examples`
                    full_definition = meaning if meaning else examples
                    redis_key = f"{word}-{idx}"  # 例如 "A.C.-0", "A.C.-1"

                    # **检查 Redis 是否已存在该词条**
                    if not r.exists(redis_key):
                        if full_definition:
                            embedding = get_embedding(full_definition)
                            r.set(redis_key, embedding.tobytes())
                            print(f"✅ 存入 Redis: {redis_key}")
                    else:
                        print(f"⚠️ 跳过 {redis_key}（已存在）")

        print(f"✅ 文件 {file} 处理完成！")


def retrieve_embedding(word_idx):
    """
    从 Redis 读取存储的 embedding 并转换回 numpy 数组
    """
    stored_embedding = r.get(word_idx)
    if stored_embedding:
        return np.frombuffer(stored_embedding, dtype=np.float32)
    else:
        print(f"⚠️ 词条 {word_idx} 不存在于 Redis")
        return None


if __name__ == "__main__":
    # json_folder = "./tmp_json"  # 词典 JSON 目录
    json_folder = "./nz_dictionary_jsons"  # 词典 JSON 目录
    process_json_files(json_folder)
    print("🎉 所有词条已存入 Redis！")

    # 示例：查询 Redis 里的某个 embedding
    test_word = "A.C.-0"  # 示例 key
    embedding_vector = retrieve_embedding(test_word)
    if embedding_vector is not None:
        print(f"🔍 Redis 中的 {test_word} 向量: {embedding_vector[:5]} ...")
