# pip install nltk pronouncing

import nltk
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


import pandas as pd
import numpy as np
import re, math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from nltk.stem import WordNetLemmatizer
import pronouncing

# 系统翻译配置
SYSTEMS = ["Google", "Reference", "B‑E System", "DeepSeek", "ChatGPT-4o"]
TRANS_ROWS = range(3, 8)
SCORE_ROWS = range(18, 23)

# === 权重配置（总和不必为1，会自动归一化）===
WEIGHTS = {
    "rhyme": 1.0,       # 押韵感
    "syllable": 1.0     # 节奏感
}

lemmatizer = WordNetLemmatizer()

# 押韵密度（rhyme）
def compute_rhyme_density(poem: str) -> float:
    lines = [line.strip() for line in poem.split('\n') if line.strip()]
    end_words = [line.split()[-1].lower() for line in lines if line.split()]
    rhyme_sets = []

    for word in end_words:
        lemma = lemmatizer.lemmatize(word)
        rhymes = set(pronouncing.rhymes(lemma))
        if not rhymes:
            rhyme_stub = word[-2:] if len(word) >= 2 else word
            rhymes = {w for w in end_words if w.endswith(rhyme_stub)}
        rhymes.add(word)
        rhyme_sets.append(rhymes)

    rhyme_pairs = 0
    total = len(end_words)
    for i in range(total):
        for j in range(i+1, total):
            if end_words[j] in rhyme_sets[i]:
                rhyme_pairs += 1
    return rhyme_pairs / total if total > 1 else 0.0

# 音节平衡度（syllable）
def compute_syllable_balance(poem: str) -> float:
    lines = [line.strip() for line in poem.split('\n') if line.strip()]
    counts = []
    for line in lines:
        count = 0
        for word in line.split():
            phones = pronouncing.phones_for_word(word.lower())
            count += pronouncing.syllable_count(phones[0]) if phones else 1
        counts.append(count)
    return 1 / (1 + np.var(counts)) if len(counts) > 1 else 1.0

# 综合诗性得分
def compute_poeticity(hyp: str) -> tuple:
    rhyme = compute_rhyme_density(hyp)
    syllable = compute_syllable_balance(hyp)
    total_w = sum(WEIGHTS.values())
    poetic = (WEIGHTS["rhyme"] * rhyme + WEIGHTS["syllable"] * syllable) / total_w
    return rhyme, syllable, poetic

# 打分写入 Excel
def evaluate_excel(excel_path="poems11.xlsx", output_path="poems11_poeticity_final.xlsx"):
    df = pd.read_excel(excel_path)
    sys_poetic_scores = {sys: [] for sys in SYSTEMS}

    for col in df.columns[1:]:
        for sys, t_row, s_row in zip(SYSTEMS, TRANS_ROWS, SCORE_ROWS):
            hyp = str(df.loc[t_row, col]).strip()
            if not hyp:
                continue
            rhyme, syllable, poetic = compute_poeticity(hyp)
            df.loc[s_row, col] = f"Poeticity {poetic:.3f} (R:{rhyme:.2f},S:{syllable:.2f})"
            sys_poetic_scores[sys].append(poetic)

    df.to_excel(output_path, index=False)
    print(f"✅ 打分写入完成 → {output_path}")
    return df, sys_poetic_scores

# 图表绘制
def plot_metric(df: pd.DataFrame, metric: str, pdf_path="6poem_poeticity_summary.pdf"):
    poems = list(df.columns[1:])
    fig_h = 3 * len(poems)
    fig, axes = plt.subplots(len(poems), 1, figsize=(10, fig_h))

    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for i, col in enumerate(poems):
        ax = axes[i]
        vals = []
        for sys, row in zip(SYSTEMS, SCORE_ROWS):
            cell = str(df.loc[row, col])
            m = re.search(r"Poeticity\s*([-0-9.]+).*R:([-0-9.]+),S:([-0-9.]+)", cell)
            if m:
                val_map = {
                    "poeticity": float(m.group(1)),
                    "rhyme": float(m.group(2)),
                    "syllable": float(m.group(3)),
                }
                vals.append(val_map[metric])
            else:
                vals.append(0.0)
        x = np.arange(len(SYSTEMS))
        ax.bar(x, vals, color="mediumseagreen")
        ax.set_xticks(x)
        ax.set_xticklabels(SYSTEMS, rotation=15)
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"{metric.capitalize()} Score")
        ax.set_title(f"Poem: {col} — {metric.capitalize()} Comparison")
        for xi, v in enumerate(vals):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)
        plt.close()
    print(f"✅ 图表已保存 → {pdf_path}")

# 主执行入口
if __name__ == "__main__":
    df, scores = evaluate_excel()
    for m in ["rhyme", "syllable", "poeticity"]:
        plot_metric(df, m, f"6poem_{m}_summary.pdf")
