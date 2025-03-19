import openai
from transformers import pipeline

#####################################
# 1. 初始化本地 Boss 模型（DeepSeek-R1-Distill-Qwen-14B）
#####################################
# 模型标识符及本地缓存目录（当前工作路径下的 "cache" 文件夹）
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
CACHE_DIR = "./cache"

print("Loading local Boss model using pipeline...")
boss_pipeline = pipeline(
    "text-generation",
    model=MODEL_ID,
    local_files_only=True,
    cache_dir=CACHE_DIR
)
print("Local Boss model loaded.")

#####################################
# 2. 配置 DeepSeek API（大模型调用）
#####################################
# 请替换为你自己的 DeepSeek API Key
DEEPSEEK_API_KEY = "sk-e5484044e6314d95b63af7f93a00ea7e"  # TODO: 替换为实际 API Key
openai.api_key = DEEPSEEK_API_KEY
# 设置 DeepSeek 的 API base（根据实际配置设置）
openai.api_base = "https://api.deepseek.com/v1"


#####################################
# 3. 定义 Boss 模型相关函数
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
    outputs = boss_pipeline(instruction, max_new_tokens=100)
    boss_prompt = outputs[0]['generated_text']
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
    outputs = boss_pipeline(instruction, max_new_tokens=150)
    review_feedback = outputs[0]['generated_text']
    return review_feedback


#####################################
# 4. 调用 DeepSeek 大模型 API 获取翻译
#####################################
def call_deepseek_api(boss_prompt: str) -> str:
    """
    使用 DeepSeek API 调用大模型，传入 Boss 模型生成的 prompt 获取翻译结果。
    """
    response = openai.ChatCompletion.create(
        model="deepseek-chat",  # 可选择 deepseek-chat 或 deepseek-reasoner
        messages=[
            {"role": "system", "content": "You are a helpful assistant specialized in translating Chinese poetry."},
            {"role": "user", "content": boss_prompt},
        ],
        temperature=0.7,
        stream=False
    )
    translation = response.choices[0].message.content
    return translation


#####################################
# 5. 多轮交互翻译流程（Demo 主函数）
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
# 6. 运行 Demo 示例
#####################################
if __name__ == "__main__":
    # 示例：用户输入需要翻译的中国古诗，并要求译文符合新西兰英语习惯
    user_query = "请把这首中国古诗翻译成符合新西兰英语习惯的诗意文本：\n《锦瑟》 - 李商隐"
    final_result = translation_workflow(user_query, max_rounds=3)
    print("=== 最终译文 ===")
    print(final_result)