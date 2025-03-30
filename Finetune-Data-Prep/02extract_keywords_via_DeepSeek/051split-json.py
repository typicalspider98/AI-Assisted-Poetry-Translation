import os
import json
def split_poem_json_with_subdir(input_path: str, output_root: str, batch_size: int = 300):
    """
    将包含多首诗歌的 JSON 文件按 batch_size 拆分为多个 JSON 文件，
    输出文件按原始 JSON 文件名创建子文件夹。
    """
    filename = os.path.splitext(os.path.basename(input_path))[0]
    sub_output_dir = os.path.join(output_root, filename)
    os.makedirs(sub_output_dir, exist_ok=True)

    with open(input_path, 'r', encoding='utf-8') as f:
        poems = json.load(f)

    total = len(poems)
    num_files = (total + batch_size - 1) // batch_size
    output_paths = []

    for i in range(num_files):
        batch = poems[i * batch_size:(i + 1) * batch_size]
        output_path = os.path.join(sub_output_dir, f"{filename}_part{i+1}.json")
        with open(output_path, 'w', encoding='utf-8') as out_f:
            json.dump(batch, out_f, ensure_ascii=False, indent=2)
        output_paths.append(output_path)

    return output_paths

# 路径
input_path = "../converted_jsons-todo/宋_1.json"
# input_path = "../converted_jsons-todo/唐.json"

output_dir = "./split_poems"
split_poem_json_with_subdir(input_path, output_dir)
