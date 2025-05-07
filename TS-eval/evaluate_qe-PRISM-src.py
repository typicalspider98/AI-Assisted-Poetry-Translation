# evaluate_qe_prism_src.py
# 自动化评估：语言模型视角 (PRISM‑src)
# ----------------------------------------------------
# 1. 逐诗逐系统计算 PRISM QE‑as‑Metric 分数 (src→hyp)
# 2. 把原始负对数似然分写回新 Excel
# 3. 将 Sigmoid 归一化后绘制：
#    • 每首诗各系统 Sigmoid 柱状图
#    • 跨全部诗歌的系统平均 Sigmoid
# ----------------------------------------------------
# 依赖：
#   ####git clone https://github.com/thompsonb/prism && cd prism && pip install -r requirements.txt && pip install .
#   pip install prism-mt
#   wget http://data.statmt.org/prism/m39v1.tar && tar -xf m39v1.tar   # MODEL_DIR=m39v1/
#   cp -r m39v1 ~/.cache/prism-mt/
# ----------------------------------------------------
import os, re, math, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---------- PRISM 初始化 ----------
try:
    from prism import Prism
except ImportError:
    sys.exit("❌ Prism 未安装，请参考脚本顶部说明安装后再运行！")

print("⏬  loading PRISM‑src model from ~/.cache/prism-mt/m39v1 …")
prism = Prism("en", temperature=1.0)

# ---------- 全局路径 ----------
EXCEL_IN  = "poems11.xlsx"
EXCEL_OUT = "54-poems11_prism_qe.xlsx"
PDF_PATH  = "54-poem_prism_summary.pdf"

SYSTEMS    = ["Google", "Reference", "B‑E System", "DeepSeek", "ChatGPT-4o"]
TRANS_ROWS = range(3, 8)
SCORE_ROWS = range(19, 24)

# ---------- 工具 ----------
def scale_sigmoid(x: float, scale: float = 3.0) -> float:
    """增强区分度的 Sigmoid 归一化"""
    return 1.0 / (1.0 + math.exp(-x / scale))

def score_prism(src: str, hyp: str) -> float:
    lp_list = prism.score(cand=[hyp], src=[src], segment_scores=True)
    return float(lp_list[0])

# ---------- 主流程 ----------
def process_excel_prism():
    df = pd.read_excel(EXCEL_IN)
    sys_raw = {s: [] for s in SYSTEMS}

    for col in df.columns[1:]:
        src = str(df.loc[0, col]).strip()
        for sys, t_row, s_row in zip(SYSTEMS, TRANS_ROWS, SCORE_ROWS):
            hyp = str(df.loc[t_row, col]).strip()
            if not hyp:
                continue
            raw = score_prism(src, hyp)
            df.loc[s_row, col] = f"PRISM {raw:.3f}"
            sys_raw[sys].append(raw)

    df.to_excel(EXCEL_OUT, index=False)
    print(f"✅ Prism 结果写回: {EXCEL_OUT}")
    return df, sys_raw

# ---------- 绘图 ----------
def plot_prism(df, sys_raw):
    poems = list(df.columns[1:])
    fig_h = 3 * len(poems) + 4
    fig, axes = plt.subplots(len(poems)+1, 1, figsize=(10, fig_h))

    for i, col in enumerate(poems):
        ax = axes[i]
        raw_vals = []
        for sys, s_row in zip(SYSTEMS, SCORE_ROWS):
            cell = str(df.loc[s_row, col])
            m = re.search(r"PRISM\s*([-0-9.]+)", cell)
            raw_vals.append(float(m.group(1)) if m else -6.0)

        # 使用增强版 Sigmoid 映射提高区分度
        sig = [scale_sigmoid(-v, scale=3.0) for v in raw_vals]
        x = np.arange(len(SYSTEMS))
        ax.bar(x, sig, color="seagreen")
        ax.set_xticks(x); ax.set_xticklabels(SYSTEMS, rotation=15)
        ax.set_ylim(0.5, 1.0)
        ax.set_yticks(np.linspace(0.5, 1.0, 6))
        ax.set_ylabel("Sigmoid")
        ax.set_title(f"Poem: {col} (PRISM‑src)")
        for xi, (sv, rv) in enumerate(zip(sig, raw_vals)):
            ax.text(xi, sv+0.01, f"{sv:.2f}\n({rv:.2f})", ha="center", va="bottom", fontsize=8)

    # 汇总图
    ax_avg = axes[-1]
    avg_sig = [scale_sigmoid(-np.mean(sys_raw[s]), scale=3.0) for s in SYSTEMS]
    x = np.arange(len(SYSTEMS))
    ax_avg.bar(x, avg_sig, color="gold")
    ax_avg.set_xticks(x); ax_avg.set_xticklabels(SYSTEMS, rotation=15)
    ax_avg.set_ylim(0.5, 1.0)
    ax_avg.set_yticks(np.linspace(0.5, 1.0, 6))
    ax_avg.set_ylabel("Avg Sigmoid")
    ax_avg.set_title("Average PRISM‑src (Sigmoid) across all Poems")
    for xi, sv in zip(x, avg_sig):
        ax_avg.text(xi, sv+0.01, f"{sv:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    with PdfPages(PDF_PATH) as pdf:
        pdf.savefig(fig)
        plt.close(fig)
    print(f"✅ 图表保存: {PDF_PATH}")

# ---------- 入口 ----------
if __name__ == "__main__":
    df_out, sys_scores = process_excel_prism()
    plot_prism(df_out, sys_scores)
