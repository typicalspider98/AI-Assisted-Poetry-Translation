import os
import json
import redis

# 连接 Redis 0号数据库
r = redis.Redis(host="localhost", port=6379, db=0)

def store_definitions_to_redis(json_folder):
    files = [f for f in os.listdir(json_folder) if f.endswith(".json")]
    print(f"📦 共找到 {len(files)} 个 JSON 文件，开始处理词条...")

    for file in files:
        file_path = os.path.join(json_folder, file)
        print(f"📥 正在读取：{file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            entries = json.load(f)

        for entry in entries:
            word = entry.get("word", "").strip()
            definitions = entry.get("definitions", [])

            for idx, definition in enumerate(definitions):
                meaning = definition.get("meaning", "").strip()
                examples = definition.get("examples", [])
                sources = definition.get("sources", [])

                key = f"{word}-{idx}"
                value = {
                    "word": word,
                    "meaning": meaning,
                    "examples": examples,
                    "sources": sources
                }

                # 避免重复写入
                if not r.exists(key):
                    r.set(key, json.dumps(value, ensure_ascii=False))
                    print(f"✅ 已写入 Redis：{key}")
                else:
                    print(f"⏩ 已存在，跳过：{key}")

    print("🎉 所有词条定义已写入 Redis（DB=0）")

if __name__ == "__main__":
    json_folder = "./nz_dictionary_jsons"  # ← 替换为你实际 JSON 存储目录
    store_definitions_to_redis(json_folder)
