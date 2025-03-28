# poem_loader.py

import json
from typing import List, Dict

def load_poems_from_json(filepath: str) -> List[Dict]:
    """
    从 JSON 文件中加载诗歌数据
    :param filepath: JSON 文件路径
    :return: 包含多个诗歌字典的列表，每首诗包含 title, dynasty, author, content 字段
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            poems = json.load(f)
        print(f"✅ 成功读取 {len(poems)} 首诗")
        return poems
    except Exception as e:
        print(f"❌ 读取诗歌文件失败: {e}")
        return []

if __name__ == "__main__":
    poems = load_poems_from_json("poems.json")
    for poem in poems:
        print(f"📜 {poem['title']} ({poem['dynasty']} · {poem['author']}):")
        print(poem["content"])
        print("-" * 40)
