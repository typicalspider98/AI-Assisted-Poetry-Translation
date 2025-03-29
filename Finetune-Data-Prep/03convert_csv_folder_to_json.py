import os
import csv
import json

def convert_each_csv_to_json(folder_path: str, output_folder: str):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            input_path = os.path.join(folder_path, filename)
            output_filename = os.path.splitext(filename)[0] + ".json"
            output_path = os.path.join(output_folder, output_filename)

            poems = []
            print(f"📥 正在处理文件: {filename}")
            with open(input_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    poem = {
                        "title": row.get("题目", "").strip(),
                        "dynasty": row.get("朝代", "").strip(),
                        "author": row.get("作者", "").strip(),
                        "content": row.get("内容", "").strip()
                    }
                    poems.append(poem)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(poems, f, ensure_ascii=False, indent=2)
            print(f"✅ 已保存 {len(poems)} 首诗到: {output_path}")

if __name__ == "__main__":
    convert_each_csv_to_json("poem_sources", "converted_jsons")
