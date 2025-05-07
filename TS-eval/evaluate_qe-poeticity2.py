
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from nltk.stem import WordNetLemmatizer
import pronouncing
import string

# 系统与行配置
SYSTEMS = ["Google", "Reference", "B‑E System", "DeepSeek", "ChatGPT-4o"]
TRANS_ROWS = range(3, 8)

lemmatizer = WordNetLemmatizer()

def normalize_word(word: str) -> str:
    return word.lower().strip(string.punctuation)

# 音素 + 拼写后缀 rhyme 评分
def rhyme_suffix_score(word1, word2):
    word1 = normalize_word(word1)
    word2 = normalize_word(word2)

    phones1 = pronouncing.phones_for_word(word1)
    phones2 = pronouncing.phones_for_word(word2)
    if phones1 and phones2:
        p1 = phones1[0].split()
        p2 = phones2[0].split()
        score = 0
        for a, b in zip(reversed(p1), reversed(p2)):
            if a == b:
                score += 1
            else:
                break
        return score / max(len(p1), len(p2))
    
    # fallback 拼写后缀匹配
    if word1[-3:] == word2[-3:]:
        return 0.6
    if word1[-2:] == word2[-2:]:
        return 0.4
    if word1[-1:] == word2[-1:]:
        return 0.2
    return 0.0

# Rhyme 分数（整首诗）
def compute_rhyme_similarity(poem: str) -> float:
    lines = [line.strip() for line in poem.split('\n') if line.strip()]
    end_words = [normalize_word(line.split()[-1]) for line in lines if line.split()]
    total = len(end_words)
    if total <= 1:
        return 0.0
    scores = []
    for i in range(total):
        for j in range(i+1, total):
            score = rhyme_suffix_score(end_words[i], end_words[j])
            scores.append(score)
    return np.mean(scores) if scores else 0.0

# 音节平衡度
def compute_syllable_balance(poem: str) -> float:
    lines = [line.strip() for line in poem.split('\n') if line.strip()]
    counts = []
    for line in lines:
        count = 0
        for word in line.split():
            phones = pronouncing.phones_for_word(word.lower().strip(string.punctuation))
            count += pronouncing.syllable_count(phones[0]) if phones else 1
        counts.append(count)
    return 1 / (1 + np.var(counts)) if len(counts) > 1 else 1.0

# 汇总图表
def plot_summary_rhyme_syllable(excel_path="poems11.xlsx", pdf_path="66poem_poeticity_hybrid_summary.pdf"):
    df = pd.read_excel(excel_path)
    rhyme_scores = {sys: [] for sys in SYSTEMS}
    syllable_scores = {sys: [] for sys in SYSTEMS}

    for col in df.columns[1:]:
        for sys, t_row in zip(SYSTEMS, TRANS_ROWS):
            hyp = str(df.loc[t_row, col]).strip()
            if not hyp:
                continue
            rhyme = compute_rhyme_similarity(hyp)
            syllable = compute_syllable_balance(hyp)
            rhyme_scores[sys].append(rhyme)
            syllable_scores[sys].append(syllable)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rhyme_means = [np.mean(rhyme_scores[sys]) for sys in SYSTEMS]
    syllable_means = [np.mean(syllable_scores[sys]) for sys in SYSTEMS]
    x = np.arange(len(SYSTEMS))

    axes[0].bar(x, rhyme_means, color="slateblue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(SYSTEMS, rotation=15)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Avg Rhyme Score")
    axes[0].set_title("System-wide Average Rhyme (Hybrid)")

    for xi, v in enumerate(rhyme_means):
        axes[0].text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)

    axes[1].bar(x, syllable_means, color="darkorange")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(SYSTEMS, rotation=15)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Avg Syllable Score")
    axes[1].set_title("System-wide Average Syllable")

    for xi, v in enumerate(syllable_means):
        axes[1].text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)
        plt.close()
    print(f"✅ 汇总图表已保存 → {pdf_path}")

# 执行主流程
if __name__ == "__main__":
    plot_summary_rhyme_syllable()
