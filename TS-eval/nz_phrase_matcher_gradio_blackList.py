import os
import json
import pickle
import re
import ahocorasick  # pip install pyahocorasick
import gradio as gr

# === 词预处理
def normalize_text_for_matching(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(\w+)[’']s\b", r"\1", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text

# === 加载 JSON 词典主词条
def get_words_from_json(json_folder="nz_dictionary_jsons"):
    all_words = set()
    for file in os.listdir(json_folder):
        if file.endswith(".json"):
            with open(os.path.join(json_folder, file), "r", encoding="utf-8") as f:
                data = json.load(f)
                for entry in data:
                    word = entry.get("word", "").strip().lower()
                    if len(word) >= 3 and word.isalpha():
                        all_words.add(word)
    return all_words

# === 黑名单加载
def load_blacklist(filepath="low_value_words.txt"):
    if not os.path.exists(filepath):
        return set()
    with open(filepath, "r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}

# === 构建与加载自动机
def build_and_save_ac_automaton(json_folder="nz_dictionary_jsons", output_path="ac_nzdict.pkl"):
    phrases = get_words_from_json(json_folder)
    A = ahocorasick.Automaton()
    for phrase in phrases:
        A.add_word(phrase, phrase)
    A.make_automaton()
    with open(output_path, "wb") as f:
        pickle.dump(A, f)

def load_ac_automaton(pkl_path="ac_nzdict.pkl"):
    with open(pkl_path, "rb") as f:
        A = pickle.load(f)
    return A

# === 主匹配函数
def match_nz_phrases(text: str, automaton, blacklist=None) -> list:
    cleaned = normalize_text_for_matching(text)
    hits = set()
    for _, phrase in automaton.iter(cleaned):
        if re.search(rf"\b{re.escape(phrase)}\b", cleaned):
            hits.add(phrase)
    if blacklist:
        hits = {w for w in hits if w not in blacklist}
    return sorted(hits)

# === 加载阶段
PKL_PATH = "ac_nzdict.pkl"
JSON_FOLDER = "nz_dictionary_jsons"
BLACKLIST_PATH = "low_value_words.txt"

if not os.path.exists(PKL_PATH):
    print("🔧 构建自动机中...")
    build_and_save_ac_automaton(JSON_FOLDER, PKL_PATH)
AC_MACHINE = load_ac_automaton(PKL_PATH)
BLACKLIST = load_blacklist(BLACKLIST_PATH)

# === Gradio 接口
def analyze_single_translation(text: str):
    filtered_hits = match_nz_phrases(text, AC_MACHINE, BLACKLIST)
    return f"🔍 命中有效词条数：{len(filtered_hits)}\n✅ 有效词条：{filtered_hits}"

with gr.Blocks() as demo:
    gr.Markdown("## 🧠 新西兰词条命中检测工具（Gradio版）")
    gr.Markdown("请输入完整翻译文本（整首诗）：")
    input_box = gr.Textbox(lines=10, label="输入翻译文本")
    output_box = gr.Textbox(lines=6, label="命中结果", interactive=False)
    analyze_button = gr.Button("分析命中词")

    analyze_button.click(fn=analyze_single_translation, inputs=input_box, outputs=output_box)

demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
