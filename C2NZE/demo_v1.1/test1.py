import openai
from openai import OpenAI  # 新版 SDK 用法
from transformers import AutoTokenizer, AutoModelForCausalLM

#####################################
# 1. 加载本地 Boss 模型（DeepSeek-R1-Distill-Qwen-14B）
#####################################
model_path = "DeepSeek-R1-Distill-Qwen-14B"  # 替换为你的本地模型路径
print("Loading local Boss model from local files...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="cpu")
print("Local Boss model loaded.")


#####################################
# 2. 配置 DeepSeek API（大模型调用）
#####################################
# 请替换为你自己的 DeepSeek API Key
DEEPSEEK_API_KEY = "sk-e5484044e6314d95b63af7f93a00ea7e"  # TODO: 替换为实际 API Key
# 使用新版 OpenAI SDK，设置 API Key 及 Base URL（新版 base_url 为 "https://api.deepseek.com"）
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

#####################################
# 3. 定义本地生成函数
#####################################
'''
使用建议
我们建议在使用 DeepSeek-R1 系列模型（包括基准测试）时遵循以下配置，以实现预期性能：
将温度设置在 0.5-0.7 范围内（建议为 0.6），以防止无休止的重复或不连贯的输出。
避免添加系统提示；所有说明都应包含在用户提示中。
对于数学问题，建议在提示中包含一个指令，例如：“请逐步推理，并将您的最终答案放在 \boxed{} 内。”
在评估模型性能时，建议进行多次测试并取平均值。
此外，我们观察到DeepSeek-R1系列模型倾向于绕过思维模式（即输出“ \n\n “），这可能会对模型的性能产生不利影响。 
为了确保模型进行彻底的推理，我们建议强制模型在每次输出开始时以“<think>\n”启动其响应。
'''
def local_generate(prompt_text: str, max_new_tokens=100, min_length=50):
    """
    使用本地模型生成文本，生成策略参数如下：
        - max_new_tokens: 最大生成 token 数（上限）
        - min_length: 最小生成 token 数，防止过早结束
        - do_sample: 启用采样生成，避免过于保守的贪心搜索
        - temperature: 控制生成随机性（0.7 适中）
        - top_p: nucleus 采样，累计概率达到 90% 的 token 集合中采样
    """
    inputs = tokenizer(prompt_text, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        min_length=min_length,
        do_sample=True,
        temperature=0.6,
        top_p=0.9
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

#####################################
# 4. 定义 Boss 模型相关函数
#####################################
def generate_boss_prompt(user_query: str) -> str:
    """
    根据用户需求生成针对大模型的 prompt，
    要求保留中国古诗意境，且译文符合新西兰英语风格。
    """
    instruction = (
        f"你是一个专业的翻译辅助系统，专门负责指导大模型生成符合新西兰英语习惯的翻译。"
        f"用户需求如下：{user_query}\n"
        f"请生成一个详细的翻译提示，要求：\n"
        f"1. 保留中文古诗的意境与韵律；\n"
        f"2. 翻译应体现新西兰英语的语言风格和文化特点；\n"
        f"请输出提示文本。"
    )
    # 调用本地生成函数，设置较高的生成上限和最小长度，保证提示内容充实
    boss_prompt = local_generate(instruction, max_new_tokens=150, min_length=100)
    return boss_prompt

def review_translation_with_boss(prompt_context: str, candidate_translation: str) -> str:
    """
    Boss 模型对大模型返回的译文进行审核，并给出修改建议或直接修正译文。
    """
    instruction = (
        f"你是专业的翻译监管系统，下面是用户需求提示：\n{prompt_context}\n"
        f"大模型翻译结果如下：\n{candidate_translation}\n"
        f"请指出译文中存在的问题，并如有需要给出改写后的版本，要求译文既保留原诗意境，又符合新西兰英语习惯。"
    )
    review_feedback = local_generate(instruction, max_new_tokens=200, min_length=100)
    return review_feedback

#####################################
# 5. 调用 DeepSeek 大模型 API 获取翻译（新版调用方式）
#####################################
def call_deepseek_api(boss_prompt: str) -> str:
    """
    使用 DeepSeek API 调用大模型，传入 Boss 模型生成的 prompt 获取翻译结果。
    这里使用新版 OpenAI SDK 客户端调用方式。
    """
    response = client.chat.completions.create(
        model="deepseek-chat",  # 可选择 deepseek-chat 或 deepseek-reasoner
        messages=[
            {"role": "system", "content": "You are a helpful assistant specialized in translating Chinese poetry."},
            {"role": "user", "content": boss_prompt},
        ],
        temperature=1.3,
        # 代码生成/数学解题0.0, 数据抽取/分析1.0, 通用对话1.3, 翻译1.3, 创意类写作/诗歌创作1.5
        stream=False
    )
    translation = response.choices[0].message.content
    return translation

#####################################
# 6. 多轮交互翻译流程（Demo 主函数）
#####################################
def translation_workflow(user_input: str, max_rounds: int = 2):
    """
    整体流程：
      1. Boss 模型根据用户需求生成 prompt；
      2. 通过 DeepSeek API 获取初步翻译；
      3. Boss 模型审核译文，如需修改则迭代生成新提示；
      4. 最终返回符合要求的译文。
    """
    print(f"\n=== 用户输入需求: {user_input} ===\n")

    # Step 1: Boss 模型生成初步 prompt
    boss_prompt = generate_boss_prompt(user_input)
    print(f"[Boss 模型] 生成的初步 Prompt:\n{boss_prompt}\n")

    final_translation = None

    for round_idx in range(max_rounds):
        print(f"--- 交互轮次 {round_idx + 1} ---")
        # Step 2: 调用 DeepSeek 大模型获得翻译
        ds_translation = call_deepseek_api(boss_prompt)
        print(f"[DeepSeek 大模型] 返回的翻译:\n{ds_translation}\n")

        # Step 3: Boss 模型审核翻译结果
        review_feedback = review_translation_with_boss(boss_prompt, ds_translation)
        print(f"[Boss 模型] 审查反馈:\n{review_feedback}\n")

        # 简单示例：若反馈中包含“改写”、“需修改”或“建议”等关键词，则迭代
        if "改写" in review_feedback or "需修改" in review_feedback or "建议" in review_feedback:
            boss_prompt = "请根据以下反馈重新生成翻译提示：" + review_feedback
        else:
            final_translation = ds_translation
            break

    if not final_translation:
        final_translation = ds_translation

    return final_translation

#####################################
# 7. 运行 Demo 示例
#####################################
if __name__ == "__main__":
    # 示例：用户输入需要翻译的中国古诗，并要求译文符合新西兰英语习惯
    user_query = (
        "请把这首中国古诗翻译成符合新西兰英语习惯的诗意文本：\n"
        "《静夜思》 - 李白\n"
        "床前明月光，疑是地上霜。举头望明月，低头思故乡。"
    )
    final_result = translation_workflow(user_query, max_rounds=3)
    print("=== 最终译文 ===")
    print(final_result)
