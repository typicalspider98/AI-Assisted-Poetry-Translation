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
    # 兼容没有 cometkiwi 时回退普通 QE
    comet_ckpt = download_model("Unbabel/wmt20-comet-qe-da")
comet_qe = load_from_checkpoint(comet_ckpt)

# -------- 超参与路径 --------
EXCEL_PATH  = "poems11.xlsx"
OUTPUT_PATH = "poems11_comet_qe3.xlsx"
PDF_PATH    = "43-poem_comet_summary.pdf"

SYSTEMS = ["Google", "Reference", "B‑E System", "DeepSeek", "ChatGPT-4o"]
TRANS_ROWS = range(3, 8)   # 源文件中五行译文
SCORE_ROWS = range(13, 18) # 写分的位置

# -------- 工具函数 --------

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def score_comet(src: str, hyp: str) -> float:
    """返回 **scalar** COMET 分。支持不同返回结构。"""
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

    # per‑system 累积列表，用于计算平均
    sys_raw_scores = {sys: [] for sys in SYSTEMS}

    for col in df.columns[1:]:  # 跳过第一列“诗名”
        src_text = str(df.loc[0, col]).strip()
        for idx, (sys, t_row, s_row) in enumerate(zip(SYSTEMS, TRANS_ROWS, SCORE_ROWS)):
            hyp = str(df.loc[t_row, col]).strip()
            if not hyp:
                continue
            raw = score_comet(src_text, hyp)
            df.loc[s_row, col] = f"COMET {raw:.3f}"
            sys_raw_scores[sys].append(raw)

    df.to_excel(out_path, index=False)
    print(f"✅ Excel 写入完成 → {out_path}")
    return df, sys_raw_scores


# -------- 绘图 --------

def plot_comet(df: pd.DataFrame, sys_raw_scores: dict, pdf_path=PDF_PATH):
    poems = list(df.columns[1:])
    num_poems = len(poems)

    fig_h = 3 * num_poems + 4  # 动态高度
    fig, axes = plt.subplots(num_poems + 1, 1, figsize=(10, fig_h))

    # ---- 逐诗图 ----
    for p_idx, col in enumerate(poems):
        ax = axes[p_idx]
        raw_vals = []
        for sys, s_row in zip(SYSTEMS, SCORE_ROWS):
            cell = str(df.loc[s_row, col])
            m = re.search(r"COMET\s*([-0-9.]+)", cell)
            raw_vals.append(float(m.group(1)) if m else -1.5)
        sig_vals = [sigmoid(v) for v in raw_vals]

        x = np.arange(len(SYSTEMS))
        ax.bar(x, sig_vals, color="steelblue")
        ax.set_xticks(x)
        ax.set_xticklabels(SYSTEMS, rotation=15)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Sigmoid")
        ax.set_title(f"Poem: {col}")
        for xi, sv in zip(x, sig_vals):
            ax.text(xi, sv + 0.01, f"{sv:.2f}", ha="center", va="bottom", fontsize=8)

    # ---- 系统平均图 ----
    ax_avg = axes[-1]
    avg_sig = [sigmoid(np.mean(sys_raw_scores[sys])) for sys in SYSTEMS]
    x = np.arange(len(SYSTEMS))
    ax_avg.bar(x, avg_sig, color="darkorange")
    ax_avg.set_xticks(x)
    ax_avg.set_xticklabels(SYSTEMS, rotation=15)
    ax_avg.set_ylim(0, 1)
    ax_avg.set_ylabel("Avg Sigmoid")
    ax_avg.set_title("Average COMET‑QE (Sigmoid) across all Poems")
    for xi, sv in zip(x, avg_sig):
        ax_avg.text(xi, sv + 0.01, f"{sv:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)
        plt.close(fig)
    print(f"✅ 图表已保存 → {pdf_path}")


# -------- 入口 --------
if __name__ == "__main__":
    df_out, sys_scores = process_excel_comet()
    plot_comet(df_out, sys_scores)
