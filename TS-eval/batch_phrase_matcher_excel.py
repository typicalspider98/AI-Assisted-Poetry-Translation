import os
import pickle
import re
import pandas as pd
import ahocorasick

# === 参数路径
EXCEL_PATH = "poems11.xlsx"
OUTPUT_PATH = "poems11_nz_check.xlsx"
AC_PKL_PATH = "ac_nzdict.pkl"
BLACKLIST_PATH = "low_value_words.txt"

# === 文本预处理
def normalize_text_for_matching(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(\w+)[’']s\b", r"\1", text)  # 单数所有格
    text = re.sub(r"\b(\w+)s[’']\b", r"\1", text)  # 复数所有格
    text = re.sub(r"[^\w\s'’-]", "", text)  # 保留撇号/连字符/空格
    return text

# === 匹配函数：返回有效词和黑名单词
def match_nz_phrases_with_blacklist(text: str, automaton, blacklist=None):
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

# === 加载自动机和黑名单
def load_ac_and_blacklist(ac_path=AC_PKL_PATH, blacklist_path=BLACKLIST_PATH):
    if not os.path.exists(ac_path):
        raise FileNotFoundError(f"❌ AC automaton file not found: {ac_path}")
    with open(ac_path, "rb") as f:
        automaton = pickle.load(f)

    blacklist = set()
    if os.path.exists(blacklist_path):
        with open(blacklist_path, "r", encoding="utf-8") as f:
            blacklist = {line.strip().lower() for line in f if line.strip()}
    return automaton, blacklist

# === 主处理函数
def process_excel_multiline(input_path, output_path):
    df = pd.read_excel(input_path)
    ac_machine, blacklist = load_ac_and_blacklist()

    for col in df.columns[1:]:  # Skip first column (e.g., index/label)
        for i, row_idx in enumerate(range(3, 8)):  # Rows 3 to 7 are translation outputs
            text = df.loc[row_idx, col] if row_idx in df.index else ""
            if isinstance(text, str) and text.strip():
                valid_hits, ignored_hits = match_nz_phrases_with_blacklist(text, ac_machine, blacklist)
                result = f"✅ Matched terms ({len(valid_hits)}): {', '.join(valid_hits)}"
                if ignored_hits:
                    result += f"\n⚠️ Blacklisted matches ({len(ignored_hits)}): {', '.join(ignored_hits)}"
            else:
                result = ""
            df.loc[8 + i, col] = result  # Write to row 8~12

    df.to_excel(output_path, index=False)
    print(f"✅ Saved results to: {output_path}")

# === 程序入口
if __name__ == "__main__":
    process_excel_multiline(EXCEL_PATH, OUTPUT_PATH)
