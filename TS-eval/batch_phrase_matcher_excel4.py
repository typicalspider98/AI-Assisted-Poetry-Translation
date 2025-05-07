import os
import pickle
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# === 参数路径
EXCEL_PATH = "poems11_nz_check.xlsx"
PDF_OUTPUT_PATH = "31-poem_nz_match_barchart_combined.pdf"  # 汇总柱状图

# === 系统名与行号
SYSTEMS = ["Google", "Reference", "B-E System", "DeepSeek", "ChatGPT-4o"]
MATCH_ROWS = range(8, 13)

# === 提取统计
def extract_counts(df):
    valid_stats = {sys: 0 for sys in SYSTEMS}
    black_stats = {sys: 0 for sys in SYSTEMS}
    cols = df.columns[1:]

    for col in cols:
        for i, row_idx in enumerate(MATCH_ROWS):
            cell = df.loc[row_idx, col]
            if not isinstance(cell, str): cell = ""
            m1 = re.search(r"Matched terms\s*\((\d+)\)", cell)
            m2 = re.search(r"Blacklisted matches\s*\((\d+)\)", cell)
            valid_stats[SYSTEMS[i]] += int(m1.group(1)) if m1 else 0
            black_stats[SYSTEMS[i]] += int(m2.group(1)) if m2 else 0

    return valid_stats, black_stats

# === 绘制合并图像
def plot_combined_barchart(valid_stats, black_stats, pdf_path):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # 图 1: 有效命中
    systems = list(valid_stats.keys())
    values = list(valid_stats.values())
    axs[0].bar(systems, values, color='green')
    axs[0].set_title("Total Valid NZ Term Matches (All Poems)")
    axs[0].set_ylabel("Total Match Count")
    axs[0].grid(axis="y", linestyle="--", alpha=0.3)
    for i, v in enumerate(values):
        axs[0].text(i, v + 0.5, str(v), ha="center", va="bottom")

    # 图 2: 黑名单命中
    systems = list(black_stats.keys())
    values = list(black_stats.values())
    axs[1].bar(systems, values, color='orange')
    axs[1].set_title("Total Blacklisted Matches (All Poems)")
    axs[1].set_ylabel("Total Match Count")
    axs[1].grid(axis="y", linestyle="--", alpha=0.3)
    for i, v in enumerate(values):
        axs[1].text(i, v + 0.5, str(v), ha="center", va="bottom")

    plt.tight_layout()
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)
        plt.close(fig)

    print(f"✅ 合并柱状图已保存到: {pdf_path}")

# === 主程序
if __name__ == "__main__":
    df = pd.read_excel(EXCEL_PATH)
    valid_stats, black_stats = extract_counts(df)
    plot_combined_barchart(valid_stats, black_stats, PDF_OUTPUT_PATH)
