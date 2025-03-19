import requests
from bs4 import BeautifulSoup
import json

# 目标 URL
url = "https://matty-plumski.github.io/nzdc-test/A/"

# 发送请求
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
response = requests.get(url, headers=headers)

# 确保请求成功
if response.status_code != 200:
    print(f"❌ 请求失败，状态码：{response.status_code}")
else:
    print("✅ 页面请求成功！")

# 解析 HTML
soup = BeautifulSoup(response.text, "html.parser")

# 存储所有词条
entries = []

# 查找所有词条（h2）
for term in soup.find_all("h2", id=True):
    word = term.contents[0].strip()  # 提取词条名称，去掉超链接等内容

    # 获取多个定义
    definitions = []
    next_tag = term.find_next_sibling()

    while next_tag and next_tag.name == "p":
        meaning = next_tag.text.strip()

        # 初始化当前定义的例句和来源
        examples = []
        sources = []

        blockquote = next_tag.find_next_sibling("blockquote")
        if blockquote:
            for p in blockquote.find_all("p"):  # 遍历所有 <p>，获取多个例句
                text = p.get_text(strip=True)

                # 查找 <i> 标签（来源）
                source_tag = p.find("i")
                source = source_tag.text.strip() if source_tag else "Unknown Source"
                if source_tag and not source.endswith(")"):
                    source += ")"

                # 移除来源部分，仅保留例句
                example = text.replace(source_tag.text, "").strip() if source_tag else text.strip()

                examples.append(example)
                sources.append(source)

        # 添加到 definitions 列表
        definitions.append({
            "meaning": meaning,
            "examples": examples,
            "sources": sources
        })

        # 继续查找下一个 <p> 作为新的定义，直到遇到新的 h2 词条
        next_tag = next_tag.find_next_sibling()
        while next_tag and next_tag.name == "p":
            meaning = next_tag.text.strip()
            blockquote = next_tag.find_next_sibling("blockquote")
            examples = []
            sources = []
            if blockquote:
                for p in blockquote.find_all("p"):
                    text = p.get_text(strip=True)
                    source_tag = p.find("i")
                    source = source_tag.text.strip() if source_tag else "Unknown Source"
                    if source_tag and not source.endswith(")"):
                        source += ")"
                    example = text.replace(source_tag.text, "").strip() if source_tag else text.strip()
                    examples.append(example)
                    sources.append(source)
            definitions.append({
                "meaning": meaning,
                "examples": examples,
                "sources": sources
            })
            next_tag = next_tag.find_next_sibling()

    # 存储数据
    entries.append({
        "word": word,
        "definitions": definitions  # 现在是一个包含多个定义的列表
    })

# 保存到 JSON
with open("nz_dictionary_A.json", "w", encoding="utf-8") as f:
    json.dump(entries, f, ensure_ascii=False, indent=4)

print("✅ A 页面爬取完成，数据已保存为 nz_dictionary_A.json！")