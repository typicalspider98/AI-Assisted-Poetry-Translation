# nz_phrase_matcher.py

import os
import json
import ahocorasick
import pickle
import re

# === 预处理翻译文本，处理所有格 + 清理标点 ===
def normalize_text_for_matching(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(\w+)[’']s\b", r"\1", text)  # 所有格去除
    text = re.sub(r"[^\w\s]", "", text)  # 去除标点
    return text


# === 从 JSON 文件夹提取主词条 ===
def get_words_from_json(json_folder="nz_dictionary_jsons"):
    all_words = set()
    for file in os.listdir(json_folder):
        if file.endswith(".json"):
            with open(os.path.join(json_folder, file), "r", encoding="utf-8") as f:
                data = json.load(f)
                for entry in data:
                    word = entry.get("word", "").strip().lower()
                    if len(word) >= 3 and word.isalpha():  # 过滤短词、无效词
                        all_words.add(word)
    return all_words


# === 构建并保存 AC 自动机 ===
def build_and_save_ac_automaton(json_folder="nz_dictionary_jsons", output_path="ac_nzdict.pkl"):
    phrases = get_words_from_json(json_folder)
    print(f"✅ 从 JSON 获取主词条数量: {len(phrases)}")

    A = ahocorasick.Automaton()
    for phrase in phrases:
        A.add_word(phrase, phrase)
    A.make_automaton()

    with open(output_path, "wb") as f:
        pickle.dump(A, f)
    print(f"✅ 自动机已保存至: {output_path}")


# === 加载持久化 AC 自动机 ===
def load_ac_automaton(pkl_path="ac_nzdict.pkl"):
    if not os.path.exists(pkl_path):
        raise FileNotFoundError("❌ 自动机文件不存在，请先构建一次！")
    with open(pkl_path, "rb") as f:
        A = pickle.load(f)
    print("✅ 自动机加载成功")
    return A


# === 匹配输入文本中命中的 NZ 短语词条 ===
def match_nz_phrases(text: str, automaton) -> list:
    cleaned = normalize_text_for_matching(text)
    hits = set()
    for _, phrase in automaton.iter(cleaned):
        if re.search(rf"\b{re.escape(phrase)}\b", cleaned):
            hits.add(phrase)
    return sorted(hits)


# === 主执行逻辑（测试入口） ===
if __name__ == "__main__":
    pkl_path = "ac_nzdict.pkl"
    json_folder = "nz_dictionary_jsons"  # 修改为你的 JSON 文件夹路径

    # 自动判断是否构建
    if not os.path.exists(pkl_path):
        print("🔧 未检测到已保存的自动机，正在初始化构建...")
        build_and_save_ac_automaton(json_folder=json_folder, output_path=pkl_path)
    else:
        print("📁 自动机文件已存在，跳过构建。")

    ac_machine = load_ac_automaton(pkl_path)

    # 示例测试文本
    # 人工
    translation1 = "The bright moon shines before my bed, I wonder if it is frost on the ground. I lift my head and look at the bright moon, I lower my head and think of my homeland."
    # google
    translation2 = "The moonlight shines brightly in front of my bed, I wonder if it is frost on the ground. I look up at the bright moon, and think of my hometown."
    # B-E
    translation3 = "Before my bed, the marama's glow, Like frost on earth, a touch of snow. I catch the moon so clear and bright, Then bow and dream of home tonight."
    # 4o
    translation4 = "Moonlight spreads across my floor— Looks just like frost upon the ground. I lift my head to gaze at the moon, Then lower it, homesick for my hometown."
    # DS
    # translation5 = "Moonlight spreads across my floor— Looks just like frost upon the ground. I lift my head to gaze at the moon, Then lower it, homesick for my hometown."

    
    matches = match_nz_phrases(translation1, ac_machine)
    print(f"🔍 命中词条数: {len(matches)}")
    print("✅ 命中词条列表:", matches)
    matches = match_nz_phrases(translation2, ac_machine)
    print(f"🔍 命中词条数: {len(matches)}")
    print("✅ 命中词条列表:", matches)
    matches = match_nz_phrases(translation3, ac_machine)
    print(f"🔍 命中词条数: {len(matches)}")
    print("✅ 命中词条列表:", matches)
    matches = match_nz_phrases(translation4, ac_machine)
    print(f"🔍 命中词条数: {len(matches)}")
    print("✅ 命中词条列表:", matches)
