import os
import json
import pickle
import re
import ahocorasick
import gradio as gr

# === 基础函数 ===
def normalize_text_for_matching(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(\w+)[’']s\b", r"\1", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text

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

def match_nz_phrases(text: str, automaton) -> list:
    cleaned = normalize_text_for_matching(text)
    hits = set()
    for _, phrase in automaton.iter(cleaned):
        if re.search(rf"\b{re.escape(phrase)}\b", cleaned):
            hits.add(phrase)
    return sorted(hits)

# === 自动机准备 ===
PKL_PATH = "ac_nzdict.pkl"
JSON_FOLDER = "nz_dictionary_jsons"
if not os.path.exists(PKL_PATH):
    print("🔧 未检测到自动机，正在构建...")
    build_and_save_ac_automaton(JSON_FOLDER, PKL_PATH)
else:
    print("✅ 自动机已存在，跳过构建。")
AC_MACHINE = load_ac_automaton(PKL_PATH)

# === Gradio UI ===
def analyze_single_translation(text: str):
    hits = match_nz_phrases(text, AC_MACHINE)
    return f"🔍 命中词条数：{len(hits)}\n✅ 命中词条：{hits}"

with gr.Blocks() as demo:
    gr.Markdown("## 🧠 新西兰本地词汇命中检测工具（单诗分析）")
    gr.Markdown("请在下方输入一首完整的翻译诗：")
    input_box = gr.Textbox(lines=10, label="输入翻译文本")
    output_box = gr.Textbox(lines=6, label="匹配结果", interactive=False)
    analyze_button = gr.Button("分析命中词条")

    analyze_button.click(fn=analyze_single_translation, inputs=input_box, outputs=output_box)

demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
