import os
import pickle
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# === 参数路径
EXCEL_PATH = "poems11_nz_check.xlsx"
PDF_OUTPUT_PATH = "21-poem_nz_match_lines_all.pdf"  # 折线图

# === 系统名与行号
SYSTEMS = ["Google", "Reference", "B-E System", "DeepSeek", "ChatGPT-4o"]
MATCH_ROWS = range(8, 13)  # 第8~12行为匹配结果

# === 解析匹配数量
def extract_counts(df):
    valid_stats = {sys: [] for sys in SYSTEMS}
    black_stats = {sys: [] for sys in SYSTEMS}
    poem_titles = []

    cols = df.columns[1:]
    for col in cols:
        poem_titles.append(col)
        for i, row_idx in enumerate(MATCH_ROWS):
            cell = df.loc[row_idx, col]
            if not isinstance(cell, str): cell = ""
            m1 = re.search(r"Matched terms\s*\((\d+)\)", cell)
            m2 = re.search(r"Blacklisted matches\s*\((\d+)\)", cell)
            valid_stats[SYSTEMS[i]].append(int(m1.group(1)) if m1 else 0)
            black_stats[SYSTEMS[i]].append(int(m2.group(1)) if m2 else 0)

    return poem_titles, valid_stats, black_stats

# === 绘制叠加图（每系统一条线）
def plot_overlayed_lines(poem_titles, valid_stats, black_stats, pdf_path):
    with PdfPages(pdf_path) as pdf:
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        x = list(range(len(poem_titles)))

        # 有效命中图
        for sys, values in valid_stats.items():
            axs[0].plot(x, values, marker="o", label=sys)
        axs[0].set_title("Valid NZ Term Matches (All Poems)")
        axs[0].set_ylabel("Count")
        axs[0].legend()
        axs[0].grid(True)

        # 黑名单图
        for sys, values in black_stats.items():
            axs[1].plot(x, values, marker="o", linestyle="--", label=sys)
        axs[1].set_title("Blacklisted Term Matches (All Poems)")
        axs[1].set_ylabel("Count")
        axs[1].set_xticks(x)
        axs[1].set_xticklabels(poem_titles, rotation=45, ha="right")
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"✅ 合并折线图已保存：{pdf_path}")

# === 主执行入口
if __name__ == "__main__":
    df = pd.read_excel(EXCEL_PATH)
    poem_titles, valid_stats, black_stats = extract_counts(df)
    plot_overlayed_lines(poem_titles, valid_stats, black_stats, PDF_OUTPUT_PATH)
