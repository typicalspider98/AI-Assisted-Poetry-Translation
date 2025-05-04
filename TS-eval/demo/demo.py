import openai
from openai import OpenAI 
import json
import os

client = OpenAI(api_key = os.environ["OPENAI_API_KEY"])

# 加载数据（每行一个 JSON 对象）
with open("data/demo.jsonl", "r", encoding="utf-8") as f:
    samples = [json.loads(line) for line in f]

for sample in samples:
    system_prompt = (
        "You are a bilingual poetry expert. Evaluate the translation quality on three dimensions:\n"
        "1. Faithfulness (Does it preserve the meaning?)\n"
        "2. Fluency (Is it grammatical and natural English?)\n"
        "3. Elegance (Does it sound poetic and stylistic?)\n"
        "Give a 1-5 score for each, and explain briefly."
    )

    user_prompt = f"""Original (Chinese):
{sample['input']}

Reference Translation:
{sample['ideal']}

Candidate Translation:
{sample['completion']}

Please respond with a JSON object like:
{{"faithfulness": 4, "fluency": 5, "elegance": 3, "comment": "..."}}
"""

    # ✅ 新 SDK 的调用方式
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    reply = response.choices[0].message.content.strip()
    print(f"\nSystem: {sample['meta']['system']}")
    print(reply)