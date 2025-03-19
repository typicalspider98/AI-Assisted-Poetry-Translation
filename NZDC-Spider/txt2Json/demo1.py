import json
import re


def parse_entry(entry_text):
    """
    解析单个词条的文本，提取相关信息，并格式化为 JSON 结构
    """
    lines = entry_text.strip().split("\n")
    word = lines[0].strip()  # 第一行是词语

    # 初始化字段
    meaning, origin = "", ""
    examples, sources = [], []

    # 解析每一行
    i = 1
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Definition:"):
            meaning = line.replace("Definition:", "").strip()
        elif line.startswith("Origin:"):
            origin = line.replace("Origin:", "").strip()
        elif line.startswith("Date:"):
            date = line.replace("Date:", "").strip()
            source, author, vol_page, quote = "", "", "", ""

            # 逐行读取相关信息，确保不会越界
            j = i + 1
            while j < len(lines) and not lines[j].startswith("Date:"):
                if lines[j].startswith("Source:"):
                    source = lines[j].replace("Source:", "").strip()
                elif lines[j].startswith("Author:"):
                    author = lines[j].replace("Author:", "").strip()
                elif lines[j].startswith("Vol / Page:"):
                    vol_page = lines[j].replace("Vol / Page:", "").strip()
                elif lines[j].startswith("Quote:"):
                    quote = lines[j].replace("Quote:", "").strip()
                j += 1

            # 格式化日期
            formatted_date = re.sub(r"^0/(\d+)/(\d+)", r"\1 \2", date)
            formatted_date = re.sub(r"^0/0/(\d+)", r"\1", formatted_date)

            # 构建 citation 格式
            source_info = f"{formatted_date},{source},{vol_page}"
            if author:
                source_info = f"{source_info} [{author}]"

            sources.append(source_info)
            if quote:
                examples.append(quote)

            # 更新索引，跳过已处理的行
            i = j - 1
        i += 1

    # 组合 meaning 信息
    if origin:
        meaning = f"abbr.{meaning} [ORIGIN: {origin}]"

    return {
        "word": word,
        "definitions": [
            {
                "meaning": meaning,
                "examples": examples,
                "sources": sources
            }
        ]
    }


def convert_txt_to_json(txt_file, json_file):
    """
    读取 txt 词典文件，并转换为 JSON 格式
    """
    with open(txt_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 以 ----- 作为词条分隔符
    entries = content.strip().split("-----")

    parsed_entries = [parse_entry(entry) for entry in entries if entry.strip()]

    # 保存为 JSON 文件
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(parsed_entries, f, ensure_ascii=False, indent=4)

    print(f"✅ 转换完成，JSON 文件已保存为 {json_file}")


# 调用转换函数
convert_txt_to_json("test.txt", "test.json")
