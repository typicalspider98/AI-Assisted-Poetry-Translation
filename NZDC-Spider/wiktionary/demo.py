import requests
from bs4 import BeautifulSoup
import json

BASE_URL = "https://en.wiktionary.org"
CATEGORY_URL = "https://en.wiktionary.org/wiki/Category:New_Zealand_English"  # ✅ 正确页面

def get_entry_links(max_links=5):
    response = requests.get(CATEGORY_URL)
    soup = BeautifulSoup(response.text, "html.parser")
    links = []

    # 从 mw-pages 区块中提取词条链接
    for div in soup.select("div.mw-category-group"):
        for a in div.find_all("a"):
            title = a.get("title")
            href = a.get("href")
            if title and href and not title.startswith("Category:"):
                links.append((title, BASE_URL + href))
            if len(links) >= max_links:
                break
        if len(links) >= max_links:
            break

    print(f"🔗 获取到 {len(links)} 个词条链接")
    return links

def extract_entry_data(title, url):
    print(f"📥 正在抓取词条: {title} => {url}")
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    entry = {
        "word": title,
        "pos": None,
        "definitions": []
    }

    # 更健壮的方式寻找 English 区块
    h2_tags = soup.find_all("h2")
    english_section = None
    for h2 in h2_tags:
        if h2.get("id") == "English" or h2.find(id="English"):
            english_section = h2
            break

    if not english_section:
        print("⚠️ 没有找到 English 区块")
        return None

    tag = english_section.find_next_sibling()
    while tag:
        if tag.name == "h2":
            break  # 到下一个语言部分，结束
        if tag.name == "h3":
            span = tag.find("span", class_="mw-headline")
            if span:
                entry["pos"] = span.text.strip().lower()
                print(f"🔍 发现词性: {entry['pos']}")
        if tag.name == "ol":
            for li in tag.find_all("li", recursive=False):
                # 移除 usage 标签
                for usage in li.select(".usage-label-sense"):
                    usage.extract()

                meaning = li.get_text(" ", strip=True)

                # 提取例句
                examples = []
                for quote in li.select(".h-quotation"):
                    ex_text = quote.get_text(strip=True)
                    if ex_text:
                        examples.append(ex_text)

                entry["definitions"].append({
                    "meaning": meaning,
                    "examples": examples,
                    "region": "New Zealand",
                    "source": "Wiktionary"
                })
                print(f"✅ 抓取定义: {meaning[:60]}...")
        tag = tag.find_next_sibling()

    if not entry["definitions"]:
        print("⚠️ 没有找到 definitions")
    return entry if entry["definitions"] else None

# 调试运行：抓取前 5 个词条及定义
sample_links = get_entry_links(5)
sample_entries = []

for title, url in sample_links:
    data = extract_entry_data(title, url)
    if data:
        sample_entries.append(data)

# 输出结果预览
for entry in sample_entries:
    print(json.dumps(entry, indent=2, ensure_ascii=False))