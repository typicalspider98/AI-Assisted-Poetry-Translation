def sort_txt_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        lines = {line.strip() for line in f if line.strip()}  # 去重 + 去空

    sorted_lines = sorted(lines, key=str.lower)  # 不区分大小写排序

    with open(output_path, "w", encoding="utf-8") as f:
        for line in sorted_lines:
            f.write(line + "\n")

    print(f"✅ 已按字典序排序并写入：{output_path}")


if __name__ == "__main__":
    input_txt = "low_value_words.txt"      # ⬅️ 修改为你的原始文件路径
    output_txt = "low_value_words.txt"    # ⬅️ 修改为输出文件名

    sort_txt_file(input_txt, output_txt)
