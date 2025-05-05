import os
import json
import pickle
import re
import ahocorasick  # pip install pyahocorasick
import gradio as gr

# === 文本预处理：保留词组结构，处理所有格
def normalize_text_for_matching(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(\w+)[’']s\b", r"\1", text)  # 单数所有格
    text = re.sub(r"\b(\w+)s[’']\b", r"\1", text)  # 复数所有格
    text = re.sub(r"\b(\w+)s[’']\b", r"\1", text)

    text = re.sub(r"[^\w\s'’-]", "", text)  # 保留撇号/连字符/空格
    return text

# === 从 JSON 文件夹提取所有合法词条（包括词组）
def get_words_from_json(json_folder="nz_dictionary_jsons"):
    all_words = set()
    for file in os.listdir(json_folder):
        if file.endswith(".json"):
            with open(os.path.join(json_folder, file), "r", encoding="utf-8") as f:
                data = json.load(f)
                for entry in data:
                    word = entry.get("word", "").strip().lower()
                    if len(word) >= 3:
                        all_words.add(word)
    return all_words

# === 加载黑名单（低价值词）
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
        return pickle.load(f)

# === 匹配主逻辑：支持词组 + 黑名单过滤
def match_nz_phrases(text: str, automaton, blacklist=None) -> tuple:
    cleaned = normalize_text_for_matching(text)
    all_hits = set()
    for _, phrase in automaton.iter(cleaned):
        if " " in phrase:
            if phrase in cleaned:
                all_hits.add(phrase)
        else:
            if re.search(rf"\b{re.escape(phrase)}\b", cleaned):
                all_hits.add(phrase)

    valid_hits = sorted(w for w in all_hits if w not in blacklist)
    ignored_hits = sorted(w for w in all_hits if w in blacklist)
    return valid_hits, ignored_hits

# === 初始化阶段
PKL_PATH = "ac_nzdict.pkl"
JSON_FOLDER = "nz_dictionary_jsons"
BLACKLIST_PATH = "low_value_words.txt"

if not os.path.exists(PKL_PATH):
    print("🔧 构建自动机中...")
    build_and_save_ac_automaton(JSON_FOLDER, PKL_PATH)

AC_MACHINE = load_ac_automaton(PKL_PATH)
BLACKLIST = load_blacklist(BLACKLIST_PATH)

# === Gradio 接口逻辑
def analyze_single_translation(text: str):
    valid_hits, ignored_hits = match_nz_phrases(text, AC_MACHINE, BLACKLIST)

    result = f"✅ 有效命中词条数：{len(valid_hits)}\n{valid_hits}"
    if ignored_hits:
        result += f"\n\n⚠️ 被黑名单过滤词条（已命中）：\n{ignored_hits}"
    return result

def refresh_blacklist_ui():
    global BLACKLIST
    BLACKLIST = load_blacklist(BLACKLIST_PATH)
    return f"✅ 黑名单已刷新，共 {len(BLACKLIST)} 项"

def show_blacklist_ui():
    if not BLACKLIST:
        return "⚠️ 当前黑名单为空"
    sorted_words = sorted(BLACKLIST)
    return f"📋 当前黑名单词条（{len(sorted_words)} 项）:\n" + ", ".join(sorted_words)

# === Gradio 前端界面
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 新西兰词条命中检测工具（支持黑名单刷新 + 词组识别）")
    gr.Markdown("请在下方输入一首完整的翻译诗：")

    input_box = gr.Textbox(lines=10, label="输入翻译文本")
    output_box = gr.Textbox(lines=6, label="命中结果", interactive=False)

    analyze_button = gr.Button("🔍 分析命中词")
    refresh_button = gr.Button("🔁 刷新黑名单")
    show_button = gr.Button("📋 查看黑名单")

    refresh_status = gr.Textbox(label="状态反馈", interactive=False, visible=True)
    blacklist_view = gr.Textbox(label="黑名单内容", lines=6, interactive=False, visible=True)

    analyze_button.click(fn=analyze_single_translation, inputs=input_box, outputs=output_box)
    refresh_button.click(fn=refresh_blacklist_ui, outputs=refresh_status)
    show_button.click(fn=show_blacklist_ui, outputs=blacklist_view)

demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
