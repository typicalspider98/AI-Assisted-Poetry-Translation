# nz_phrase_matcher.py

import os
import json
import ahocorasick
import pickle
import re
import pandas as pd


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


def process_translation_table(filepath, ac_machine):
    df = pd.read_excel(filepath) if filepath.endswith(".xlsx") else pd.read_csv(filepath, encoding='gbk')
    poems = df.columns.tolist()
    results = {}

    for col in df.columns:
        poem_name = col
        versions = df[col].dropna().tolist()  # 第一行是标题，后面是翻译版本

        version_results = []
        for i, version in enumerate(versions):
            hits = match_nz_phrases(version, ac_machine)
            version_results.append({
                "version_index": i + 1,
                "hit_count": len(hits),
                "hits": hits
            })

        results[poem_name] = version_results

    return results


if __name__ == "__main__":
    pkl_path = "ac_nzdict1.pkl"
    json_folder = "nz_dictionary_jsons"
    input_table = "translation_versions1.csv"  # <-- 

    # 自动构建或加载自动机
    if not os.path.exists(pkl_path):
        print("🔧 未检测到已保存的自动机，正在初始化构建...")
        build_and_save_ac_automaton(json_folder=json_folder, output_path=pkl_path)
    else:
        print("📁 自动机文件已存在，跳过构建。")

    ac_machine = load_ac_automaton(pkl_path)

    # 执行表格处理
    results = process_translation_table(input_table, ac_machine)

    # 打印结果
    for poem, versions in results.items():
        print(f"\n📜 诗歌: {poem}")
        for res in versions:
            print(f"  版本{res['version_index']} 命中: {res['hit_count']} 个词条 → {res['hits']}")