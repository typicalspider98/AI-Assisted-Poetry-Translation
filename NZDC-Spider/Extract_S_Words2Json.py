import json
import re
from collections import defaultdict


def parse_entries(content):
    """
    解析多个词条，支持一词多义合并，并格式化为 JSON 结构
    """
    entries = defaultdict(list)
    current_word = None
    meaning, origin, register = "", "", ""
    examples, sources = [], []

    lines = content.strip().split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # 识别新的单词词条（如果当前行为非空，且前一行为空行，且非 Date: 开头，则为新词）
        if line and (current_word is None or (i > 0 and lines[i - 1].strip() == "" and not line.startswith("Date:"))):
            if current_word and (meaning or examples or sources):  # 存储上一个单词的定义
                entries[current_word].append({
                    "meaning": f"abbr.{meaning} [ORIGIN: {origin}{', ' + register if register else ''}]" if meaning else "",
                    "examples": examples,
                    "sources": sources
                })

            current_word = line  # 更新当前单词
            meaning, origin, register = "", "", ""
            examples, sources = [], []

        elif line.startswith("Definition:"):
            if meaning or examples or sources:  # 存储已有定义（避免空的 Definition 覆盖已有数据）
                entries[current_word].append({
                    "meaning": f"abbr.{meaning} [ORIGIN: {origin}{', ' + register if register else ''}]" if meaning else "",
                    "examples": examples,
                    "sources": sources
                })
                meaning, origin, register = "", "", ""
                examples, sources = [], []
            meaning = line.replace("Definition:", "").strip()
        elif line.startswith("Origin:"):
            origin = line.replace("Origin:", "").strip()
        elif line.startswith("Notes:") and "attrib." in line:
            register = "attrib."
        elif line.startswith("Date:"):
            date = line.replace("Date:", "").strip()
            source, author, vol_page, quote = "", "", "", ""

            j = i + 1
            while j < len(lines) and lines[j].strip():  # 直到遇到新的空行
                if lines[j].startswith("Source:"):
                    source = lines[j].replace("Source:", "").strip()
                elif lines[j].startswith("Author:"):
                    author = lines[j].replace("Author:", "").strip()
                elif lines[j].startswith("Vol / Page:"):
                    vol_page = lines[j].replace("Vol / Page:", "").strip()
                elif lines[j].startswith("Quote:"):
                    quote = lines[j].replace("Quote:", "").strip()
                j += 1

            formatted_date = re.sub(r"^0/(\d+)/(\d+)", r"\1 \2", date)
            formatted_date = re.sub(r"^0/0/(\d+)", r"\1", formatted_date)

            source_info = f"{formatted_date},{source},{vol_page}"
            if author:
                source_info = f"{source_info} [{author}]"

            sources.append(source_info)
            if quote:
                examples.append(quote)

            i = j - 1
        i += 1

    # 存入最后一个单词的定义（确保最后的词条不会丢失，即使 Definition 为空）
    if current_word and (meaning or examples or sources):
        entries[current_word].append({
            "meaning": f"abbr.{meaning} [ORIGIN: {origin}{', ' + register if register else ''}]" if meaning else "",
            "examples": examples,
            "sources": sources
        })

    return [{"word": word, "definitions": definitions} for word, definitions in entries.items()]


def convert_txt_to_json(txt_file, json_file):
    """
    读取 txt 词典文件，并转换为 JSON 格式，支持多个词的识别，并合并一词多义
    """
    with open(txt_file, "r", encoding="utf-8") as f:
        content = f.read()

    parsed_entries = parse_entries(content)

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(parsed_entries, f, ensure_ascii=False, indent=4)

    print(f"✅ 转换完成，JSON 文件已保存为 {json_file}")


# 调用转换函数
convert_txt_to_json("s_words.txt", "s_word.json")