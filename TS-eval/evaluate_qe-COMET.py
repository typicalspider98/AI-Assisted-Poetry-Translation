# evaluate_qe_comet_only.py
# 自动化评估：仅使用 COMET‑QE（wmt22‑cometkiwi‑da）
# 1. 逐诗、逐系统打分并写回 Excel（保留原始分）
# 2. 画两张图：
#    • 每首诗的各系统 Sigmoid 分布
#    • 各系统跨全部诗歌的平均 Sigmoid 分

import os, re, math, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from comet import download_model, load_from_checkpoint

# -------- 模型加载 --------
print("⏬  loading COMET‑Kiwi (wmt22)…")
try:
    comet_ckpt = download_model("Unbabel/wmt22-cometkiwi-da")
except Exception:
    comet_ckpt = download_model("Unbabel/wmt20-comet-qe-da")
comet_qe = load_from_checkpoint(comet_ckpt)

# -------- 路径配置 --------
EXCEL_PATH  = "poems11.xlsx"
OUTPUT_PATH = "poems11_comet_qe3.xlsx"
PDF_PATH    = "4poem_comet_summary-COMET.pdf"

SYSTEMS = ["Google", "Reference", "B‑E System", "DeepSeek", "ChatGPT-4o"]
TRANS_ROWS = range(3, 8)
SCORE_ROWS = range(13, 18)

# -------- 工具函数 --------
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def inverse_sigmoid(x: float) -> float:
    """1 - Sigmoid，用于表达‘跳脱原文’程度"""
    return 1.0 - sigmoid(x)

def score_comet(src: str, hyp: str) -> float:
    data = [{"src": src, "mt": hyp}]
    preds = comet_qe.predict(data, gpus=1 if torch.cuda.is_available() else 0, batch_size=8)
    p0 = preds[0]
    if isinstance(p0, dict) and "score" in p0:
        return float(p0["score"])
    if isinstance(p0, (float, int)):
        return float(p0)
    if isinstance(p0, (list, tuple)) and p0:
        return float(np.mean(p0))
    raise TypeError(f"Unsupported prediction type: {type(p0)} → {p0}")

# -------- 主流程 --------
def process_excel_comet(src_path=EXCEL_PATH, out_path=OUTPUT_PATH):
    df = pd.read_excel(src_path)
    sys_raw_scores = {sys: [] for sys in SYSTEMS}

    for col in df.columns[1:]:
        src_text = str(df.loc[0, col]).strip()
        for sys, t_row, s_row in zip(SYSTEMS, TRANS_ROWS, SCORE_ROWS):
            hyp = str(df.loc[t_row, col]).strip()
            if not hyp:
                continue
            raw = score_comet(src_text, hyp)
            df.loc[s_row, col] = f"COMET {raw:.3f}"
            sys_raw_scores[sys].append(raw)

    df.to_excel(out_path, index=False)
    print(f"✅ Excel 写入完成 → {out_path}")
    return df, sys_raw_scores

# -------- 反向图绘制（Creative Score）--------
def plot_comet(df: pd.DataFrame, sys_raw_scores: dict, pdf_path=PDF_PATH):
    poems = list(df.columns[1:])
    num_poems = len(poems)
    fig_h = 3 * num_poems + 4
    fig, axes = plt.subplots(num_poems + 1, 1, figsize=(10, fig_h))

    for p_idx, col in enumerate(poems):
        ax = axes[p_idx]
        raw_vals = []
        for sys, s_row in zip(SYSTEMS, SCORE_ROWS):
            cell = str(df.loc[s_row, col])
            m = re.search(r"COMET\s*([-0-9.]+)", cell)
            raw_vals.append(float(m.group(1)) if m else -1.5)

        inv_vals = [inverse_sigmoid(v) for v in raw_vals]
        x = np.arange(len(SYSTEMS))
        ax.bar(x, inv_vals, color="seagreen")
        ax.set_xticks(x)
        ax.set_xticklabels(SYSTEMS, rotation=15)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Creative Score")
        ax.set_title(f"Poem: {col} — Expressive Divergence (1 - COMET Sigmoid)")

        for xi, (iv, rv) in enumerate(zip(inv_vals, raw_vals)):
            ax.text(xi, iv + 0.01, f"{iv:.2f}\n({rv:.2f})", ha="center", va="bottom", fontsize=8)

    # ---- 平均图 ----
    ax_avg = axes[-1]
    avg_inv = [inverse_sigmoid(np.mean(sys_raw_scores[sys])) for sys in SYSTEMS]
    x = np.arange(len(SYSTEMS))
    ax_avg.bar(x, avg_inv, color="goldenrod")
    ax_avg.set_xticks(x)
    ax_avg.set_xticklabels(SYSTEMS, rotation=15)
    ax_avg.set_ylim(0, 1)
    ax_avg.set_ylabel("Avg Creative Score")
    ax_avg.set_title("Average Expressive Divergence (1 - COMET-QE Sigmoid)")

    for xi, iv in zip(x, avg_inv):
        ax_avg.text(xi, iv + 0.01, f"{iv:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)
        plt.close(fig)
    print(f"✅ 图表已保存（Creative Score）→ {pdf_path}")

# -------- 入口 --------
if __name__ == "__main__":
    df_out, sys_scores = process_excel_comet()
    plot_comet(df_out, sys_scores)
