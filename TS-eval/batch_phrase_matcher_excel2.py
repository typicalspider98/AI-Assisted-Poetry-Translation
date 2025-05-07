import os
import pickle
import re
import pandas as pd
import ahocorasick
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

# === 参数路径
EXCEL_PATH = "poems11.xlsx"
OUTPUT_PATH = "poems11_nz_check.xlsx"
PDF_OUTPUT_PATH = "11-poem_nz_match_summary.pdf"
AC_PKL_PATH = "ac_nzdict_new.pkl"
BLACKLIST_PATH = "low_value_words.txt"

# === 文本清洗
def normalize_text_for_matching(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(\w+)[’']s\b", r"\1", text)
    text = re.sub(r"\b(\w+)s[’']\b", r"\1", text)
    text = re.sub(r"[^\w\s'’-]", "", text)
    return text

# === 匹配函数
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

# === 加载词典与黑名单
def load_ac_and_blacklist(ac_path=AC_PKL_PATH, blacklist_path=BLACKLIST_PATH):
    with open(ac_path, "rb") as f:
        automaton = pickle.load(f)
    blacklist = set()
    if os.path.exists(blacklist_path):
        with open(blacklist_path, "r", encoding="utf-8") as f:
            blacklist = {line.strip().lower() for line in f if line.strip()}
    return automaton, blacklist

# === 处理 Excel 文件并写入检测结果
def process_excel_multiline(input_path, output_path):
    df = pd.read_excel(input_path)
    ac_machine, blacklist = load_ac_and_blacklist()

    for col in df.columns[1:]:
        for i, row_idx in enumerate(range(3, 8)):
            text = df.loc[row_idx, col] if row_idx in df.index else ""
            if isinstance(text, str) and text.strip():
                valid_hits, ignored_hits = match_nz_phrases_with_blacklist(text, ac_machine, blacklist)
                result = f"Matched terms ({len(valid_hits)}): {', '.join(valid_hits)}"
                if ignored_hits:
                    result += f"\nBlacklisted matches ({len(ignored_hits)}): {', '.join(ignored_hits)}"
            else:
                result = ""
            df.loc[8 + i, col] = result
    df.to_excel(output_path, index=False)
    print(f"✅ Excel results saved to: {output_path}")
    return df

# === 生成单页图表 PDF
def generate_summary_pdf(df, pdf_path):
    systems = ["Google", "Reference", "B-E System", "DeepSeek", "ChatGPT-4o"]
    match_rows = range(8, 13)
    cols = df.columns[1:]

    fig, axes = plt.subplots(len(cols), 2, figsize=(14, 3 * len(cols)))
    for row_idx, col in enumerate(cols):
        valid_counts, black_counts = [], []
        for r in match_rows:
            cell = df.loc[r, col]
            m1 = re.search(r"Matched terms\s*\((\d+)\)", cell if isinstance(cell, str) else "")
            m2 = re.search(r"Blacklisted matches\s*\((\d+)\)", cell if isinstance(cell, str) else "")
            valid_counts.append(int(m1.group(1)) if m1 else 0)
            black_counts.append(int(m2.group(1)) if m2 else 0)

        ax_valid = axes[row_idx][0]
        ax_black = axes[row_idx][1]
        ax_valid.bar(systems, valid_counts, color="green")
        ax_black.bar(systems, black_counts, color="orange")

        ax_valid.set_title(f"✅ Valid Terms: {col}")
        ax_black.set_title(f"⚠️ Blacklisted: {col}")
        ax_valid.set_ylim(0, max(valid_counts + [1]) + 1)
        ax_black.set_ylim(0, max(black_counts + [1]) + 1)

    plt.tight_layout()
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)
        plt.close(fig)
    print(f"✅ Summary PDF saved to: {pdf_path}")

# === 主程序
if __name__ == "__main__":
    df_out = process_excel_multiline(EXCEL_PATH, OUTPUT_PATH)
    generate_summary_pdf(df_out, PDF_OUTPUT_PATH)
