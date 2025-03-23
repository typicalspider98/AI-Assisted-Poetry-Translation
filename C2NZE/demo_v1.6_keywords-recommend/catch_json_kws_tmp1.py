import re

text = """
好的，我现在需要帮用户分析《静夜思》这首诗，并从中提取5到8个关键词或意象短语，用于指导英文翻译。
</think>

```json
{
  "keywords": [
    "bed",
    "bright moonlight",
    "dew on the ground",
    "gaze at the moon",
    "yearning for home"
  ]
}
"""

match = re.search(r"```json\s*({[\s\S]*?})\s*", text)

if match: 
    print("✅ 提取成功，内容为：") 
    print(match.group(1)) 
else: 
    print("❌ 没有提取到 JSON")