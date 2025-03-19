import requests
from bs4 import BeautifulSoup
import json
import string

# 发送请求的通用头部
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# 遍历 A-Z 的网页
import os

# 创建存放 JSON 文件的文件夹
output_dir = "nz_dictionary_jsons"
os.makedirs(output_dir, exist_ok=True)

for letter in string.ascii_uppercase:
    url = f"https://matty-plumski.github.io/nzdc-test/{letter}/"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"❌ {letter} 页面请求失败，状态码：{response.status_code}")
        continue
    else:
        print(f"✅ {letter} 页面请求成功！")

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
            meaning = next_tag.get_text(strip=True)

            # 查找与当前定义关联的例句
            examples = []
            sources = []
            blockquote = next_tag.find_next_sibling("blockquote")
            if blockquote:
                for p in blockquote.find_all("p"):  # 遍历所有 <p>，获取多个例句
                    text = ' '.join(['[' + t.get_text(strip=True) + ']' if t.name == 'b' else t.get_text(
                        strip=True) if hasattr(t, 'get_text') else t for t in p.contents])

                    # 查找 <i> 标签（来源）
                    source_tags = p.find_all("i")
                    source_tag = source_tags[-1] if source_tags else None
                    source = source_tag.get_text(strip=True).strip('()') if source_tag else "Unknown Source"

                    # 移除来源部分，仅保留例句
                    example = text.replace(source_tag.get_text(strip=True), "").strip().rstrip(
                        ') ').rstrip() if source_tag else text.strip().rstrip(') ').rstrip()

                    examples.append(example)
                    sources.append(source)

            # 添加到 definitions 列表
            definitions.append({
                "meaning": meaning,
                "examples": examples,
                "sources": sources
            })

            # 继续查找下一个 <p> 作为新的定义，直到遇到新的 h2 词条
            next_tag = blockquote.find_next_sibling() if blockquote else next_tag.find_next_sibling()

        # 存储数据
        entries.append({
            "word": word,
            "definitions": definitions  # 现在是一个包含多个定义的列表
        })

    # 保存到 JSON
    filename = os.path.join(output_dir, f"nz_dictionary_{letter}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=4)

    print(f"✅ {letter} 页面爬取完成，数据已保存为 {filename}！")
