import sys
import openai
import torch
from openai import OpenAI  # 新版 SDK 用法
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import os
from transformers import BitsAndBytesConfig

# 限制 PyTorch 的分配策略，以减少显存碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# translation_logic.py 顶部加入（或放入专用模块也行）
import redis
import sys

def check_redis_connection_or_exit(host="localhost", port=6379, db=0):
    try:
        r = redis.Redis(host=host, port=port, db=db)
        r.ping()
        print(f"✅ Redis 数据库 {db} 已连接")
    except redis.ConnectionError:
        print("❌ Redis 数据库连接失败，请检查 Redis 服务是否已启动。"
        "\n Quick launch:\nnohup dockerd > dockerd.log 2>&1 &\n"
        "docker start redis-server_NZDdictionary")
        sys.exit("❌ 项目启动终止：Redis 数据库不可用。")

check_redis_connection_or_exit(db=0)
check_redis_connection_or_exit(db=2)

global custom_model_path, tokenizer, model
#####################################
# 日志配置：创建 logs 文件夹，并生成日志文件
#####################################
if not os.path.exists("logs"):
    os.makedirs("logs")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join("logs", f"output_{timestamp}.md")
log_file = open(log_filename, "w", encoding="utf-8")


def write_log(entry: str):
    """记录日志到文件，附加时间戳"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file.write(f"[{ts}] {entry}\n\n")
    log_file.flush()


# 用于存储用户指定的本地模型路径和 Token
custom_model_path = None
local_model_token = None

# 用 accelerate 让 PyTorch 逐步加载模型
from accelerate import infer_auto_device_map

def set_model_path(new_path: str):
    global custom_model_path, tokenizer, model
    """ 设置本地模型路径 """
    new_path = os.path.abspath(new_path)
    custom_model_path = new_path
    write_log(f"用户设置了新的模型路径: {new_path}")
    print(f"✅ [DEBUG] 模型路径已更新: {new_path}")  # 方便调试
    # return f"路径已确认: {new_path}"  # ✅ 让 Gradio UI 显示状态信息
    # 加载新模型
    try:
        write_log("开始加载本地 Boss 模型...")
        print("Loading local Boss model from local files...")
        # model_path = "../DeepSeek-R1-Distill-Llama-8B"  # 请替换为你的本地模型路径
        # model_path = "../DeepSeek-R1-Distill-Qwen-14B"  # 请替换为你的本地模型路径
        # model_path = "../Qwen2.5-7B-Instruct"  # 请替换为你的本地模型路径
        default_model_path = os.path.abspath("/workspace/Project-Code/AI-Assisted-Poetry-Translation/C2NZE/models/DeepSeek-R1-Distill-Qwen-14B")
        model_path = custom_model_path if custom_model_path else default_model_path
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

        #model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        # device_map = infer_auto_device_map(model, max_memory={0: "14GB", "cpu": "8GB"})  
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, local_files_only=True,  # load_in_8bit=True,  # 启用 8-bit 量化
                                                    device_map="auto",  # device_map="auto",
                                                    torch_dtype=torch.float16,
                                                    quantization_config=BitsAndBytesConfig(
                                                        load_in_4bit=True,
                                                        bnb_4bit_compute_dtype=torch.float16,
                                                        bnb_4bit_quant_type="nf4",
                                                        )
                                                    )
                                                    # torch_dtype="auto")  # 半精度计算），节省显存
        # model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="cpu")
        print("Local Boss model loaded.")
        write_log("本地 Boss 模型加载完成。")
        return f"✅ 模型加载成功: {new_path}"  # 返回成功信息
    except Exception as e:
        write_log(f"❌ 模型加载失败: {str(e)}")
        return f"❌ 模型加载失败: {str(e)}"


def set_local_model_token(token: str):
    """ 设置本地模型 Token """
    global local_model_token
    local_model_token = token
    write_log(f"用户设置了新的本地模型 Token")
    print(f"✅ [DEBUG] 模型token已更新: {token}")  # 方便调试
    return f"路径已确认: {local_model_token}"  # ✅ 让 Gradio UI 显示状态信息


#####################################
# 1. 加载本地 Boss 模型（DeepSeek-R1-Distill-Qwen-14B）
#####################################


#####################################
# 2. 配置 DeepSeek API（大模型调用）
#####################################
# 请替换为你自己的 DeepSeek API Key
DEEPSEEK_API_KEY = "sk-e5484044e6314d95b63af7f93a00ea7e"  # TODO: 替换为实际 API Key
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
write_log("DeepSeek API 配置完成。")


#####################################
# 3. 定义本地生成函数
#####################################
def local_generate(prompt_text: str, max_new_tokens=128, min_length=50):
    global custom_model_path, tokenizer, model
    """
    使用本地模型生成文本
    参数说明：
        - max_new_tokens: 最大生成 token 数
        - min_length: 最小生成 token 数，防止过早结束
        - do_sample: 启用采样，避免贪心搜索
        - temperature: 控制生成随机性（推荐 0.6）
        - top_p: nucleus 采样（例如 0.9）
    """
    write_log(f"本地生成调用，输入 prompt:\n{prompt_text}")
    if model is None or tokenizer is None:
        return "⚠️ 还未加载模型，请先设置模型路径！"

    # 在 Prompt 头部显式加入 <think>\n 以强制思维模式
    # modified_prompt = "<think>\n" + prompt_text
    # inputs = tokenizer(modified_prompt, return_tensors="pt")

    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")  # for GPU
    # inputs = tokenizer(prompt_text, return_tensors="pt")  # for CPU
    print("max_new_tokens:", max_new_tokens)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        min_length=min_length,
        do_sample=True,
        # temperature=0.8,
        # top_p=0.9
        temperature=0.6,
        top_p=0.85,
        repetition_penalty=1.1,  # 抑制重复
        eos_token_id=tokenizer.eos_token_id  # 显式指定结束符
    )
    '''
    不让 result 包含 prompt_text，只保留新生成的内容
    '''
    input_length = inputs["input_ids"].shape[-1]
    generated_tokens = outputs[0][input_length:]
    result = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # 删除</think> 及其前面的内容（如果存在）
    if "</think>" in result:
        result = result.split("</think>")[-1].strip()
    # result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    write_log(f"本地生成返回结果:\n{result}")
    return result


#####################################
# 新增：生成初始提示文本（instruction）的函数
#####################################
def generate_instruction_text(user_query: str) -> str:
    """
    根据用户输入诗句生成初始提示文本，供用户查看和编辑。
    instruction = (
        f"你是一个专业的翻译辅助系统，专门负责指导大模型生成符合新西兰英语习惯的翻译。"
        f"用户需求如下：{user_query}\n"
        f"请生成一个详细的翻译提示，要求：\n"
        f"1. 保留中文古诗的意境与韵律；\n"
        f"2. 翻译应体现新西兰英语的语言风格和文化特点；\n"
        f"请直接给出prompt用于指导大模型进行上面诗歌的翻译工作（包括诗歌内容、如何翻译和注意事项）。"
    )
    """
    instruction = (
        f"你是一个专业的翻译辅助系统，专门负责指导大模型生成符合新西兰地区英语使用习惯的翻译。"
        f"用户需要翻译的内容是：\n{user_query}\n"
        f"请你直接给出一个中文prompt，用于指导大模型进行上述中文诗歌的英文翻译工作。\n"
        f"生成的中文prompt要求：\n"
        f"1.prompt中首先指出用户提供的中文诗歌原内容，并提出将所给中文诗歌翻译成英文诗歌这个要求。\n"
        f"2.prompt应参考用户给出的中文诗歌的具体内容，选择恰当的语言描述本首诗歌在翻译过程中应注意的事项。"
        f"（例如：尽量传达原诗的核心思想、情感和主旨，避免曲解或过度延伸；"
        f"保留诗歌中的文化意象，并选择合适的英文表达；语言要有韵味，保持诗性表达，如节奏、押韵（若能做到）、简练、修辞等；"
        f"对一些特有的文化典故、历史人物或风俗需做注解、意译或文化转化，避免读者误解；根据具体诗句灵活取舍，重意境时意译，重结构或修辞时偏直译）\n"
        f"3.prompt应指出在翻译时要体现新西兰地区英语的语言风格和文化特点。"
        f"（关于什么是新西兰英语，我指的是下面可能会提供给你的一部分关键词，这些关键词取自由一本新西兰英语词典生成的向量数据库。"
        f"在翻译本诗的过程中，有些英文单词在新西兰英语中有特殊的表达，我们将这些特殊的使用情况从向量数据库中取出来，"
        f"以关键词及例句的形式展示，你在生成prompt时可以参考这些词的使用场景，看是否可以将其融入你的诗歌翻译中。"
        f"要尽可能保留诗歌的原有含义及意境，同时兼具新西兰英语的表达特色。）"
    )
    write_log(f"生成初始提示文本:\n{instruction}")
    return instruction
def generate_instruction_text_EN(user_query: str) -> str:
    """
    根据用户输入诗句生成初始提示文本，供用户查看和编辑。
    instruction = (
        f"你是一个专业的翻译辅助系统，专门负责指导大模型生成符合新西兰英语习惯的翻译。"
        f"用户需求如下：{user_query}\n"
        f"请生成一个详细的翻译提示，要求：\n"
        f"1. 保留中文古诗的意境与韵律；\n"
        f"2. 翻译应体现新西兰英语的语言风格和文化特点；\n"
        f"请直接给出prompt用于指导大模型进行上面诗歌的翻译工作（包括诗歌内容、如何翻译和注意事项）。"
    )
    """
    instruction = (
        f"You are a professional translation support system, specifically designed to generate prompts "
        f"that guide large language models in producing translations that align with the usage and stylistic features of New Zealand English."
        f"The user needs to translate the following text:\n{user_query}\n"
        f"Please directly generate a prompt that instructs the model to translate the above Chinese poem into English poetry."
        f"The requirements for the generated prompt are as follows (do not include the word **Prompt:**   in the output):\n"
        f"1. The prompt must be written in English.\n"
        f"2. The first sentence should clearly instruct the model to translate the provided Chinese poem into an English poem, "
        f"and it must include the original Chinese text.\n"
        f"3. The prompt should take into account the specific content of the poem and describe appropriate considerations for the translation process. "
        f"For example:\n"
        f"Strive to convey the core ideas, emotions, and main message of the original poem, avoiding distortion or excessive elaboration."
        f"Preserve the cultural imagery within the poem and choose suitable English expressions."
        "Maintain poetic qualities such as rhythm, rhyme (if feasible), conciseness, and literary devices."
        "When dealing with specific cultural references, historical figures, or customs, include annotation, "
        "cultural adaptation, or interpretive translation to prevent misunderstanding.\n"
        "Adapt the translation approach flexibly according to each line—favoring interpretive translation for imagery-rich lines, "
        "and more direct translation for structurally or rhetorically important lines.\n"
        "The prompt should also emphasize the need to reflect the linguistic style and cultural characteristics of New Zealand English in the translation.\n"
        "(Regarding what New Zealand English entails, the user may provide a list of keywords derived from a vector database "
        "built from a New Zealand English dictionary. In some cases, certain English words may have unique usages in New Zealand English. "
        "We extract these usage cases from the database and present them as keywords with sample sentences. "
        "When generating the prompt, you may refer to these usage contexts to see whether they can be incorporated into the poem's translation. The goal is to retain the original poem's meaning and imagery as much as possible, while also capturing the distinctive qualities of New Zealand English.)"
    )
    write_log(f"生成初始EN提示文本:\n{instruction}")
    return instruction

# 把不需要的 chain-of-thought 标记过滤掉
def clean_prompt(prompt: str) -> str:
    # 移除 chain-of-thought 标记，比如 <think> 和 </think>
    prompt = prompt.replace("<think>", "").replace("</think>", "")
    return prompt.strip()


def call_local_qwen_with_instruction(instruction: str, max_new_tokens=128, min_length=100) -> str:
    """
    使用本地 Qwen 模型生成最终的 prompt0，
    这里的输入为用户编辑后的提示文本（instruction）。
    """
    write_log(f"使用用户编辑后的 instruction 调用本地 Qwen 模型，输入:\n{instruction}")
    if not instruction.strip():
        write_log("❌ [ERROR] instruction 为空，无法执行本地 Qwen")
        return "⚠️ 提示文本为空，请先生成或输入提示文本！"
    # 确保 max_new_tokens 是整数
    try:
        max_new_tokens = int(max_new_tokens)
        min_length = int(min_length)
    except ValueError:
        return "⚠️ max_new_tokens 或 min_length 不是整数，请检查输入！"

    # 强制思考模式，并要求直接生成提示文本
    # formatted_instruction = f"<think>\n{instruction}\n</think>\n请直接生成用于 DeepSeek API 的翻译提示，不要包含解释或推理过程。"
    # result = local_generate(formatted_instruction, max_new_tokens=max_new_tokens, min_length=min_length)
    result = local_generate(instruction, max_new_tokens=max_new_tokens, min_length=min_length)
    # result = clean_prompt(result)  # 清洗不需要的标记
    write_log(f"本地 Qwen 模型返回的 prompt0:\n{result}")
    return result


#####################################
# 4. 定义 Boss 模型相关函数（后续交互中用于审核）
#####################################
def review_translation_with_boss(prompt_context: str, candidate_translation: str) -> str:
    """
    Boss 模型对大模型返回的译文进行审核，并给出修改建议或直接修正译文。
    """
    instruction = (
        f"你是专业的翻译监管系统，下面是用户需求提示：\n{prompt_context}\n"
        f"大模型翻译结果如下：\n{candidate_translation}\n"
        f"请指出译文中存在的问题，并如有需要给出改写后的版本，要求译文既保留原诗意境，又符合新西兰英语习惯。\n"
        "只需给出翻译问题分析与改进建议，无需给出改后译文和修改说明"
    )
    review_feedback = local_generate(instruction, max_new_tokens=4000, min_length=100)
    write_log(
        f"审核翻译调用：\n使用 prompt:\n{prompt_context}\n候选翻译:\n{candidate_translation}\n审核反馈:\n{review_feedback}")
    return review_feedback


#####################################
# 5. DeepSeek API 调用函数
#####################################
def call_deepseek_api(boss_prompt: str) -> str:
    """
    使用 DeepSeek API 调用大模型，传入 Boss 模型生成的 prompt 获取翻译结果。
    """
    write_log(f"调用 DeepSeek API，请求内容:\n{boss_prompt}")
    response = client.chat.completions.create(
        model="deepseek-chat",  # 或 "deepseek-reasoner"
        messages=[
            {"role": "system", "content": "You are a helpful assistant specialized in translating Chinese poetry."},
            {"role": "user", "content": boss_prompt},
        ],
        temperature=1.4,
        stream=False
    )
    translation = response.choices[0].message.content
    write_log(f"DeepSeek API 返回翻译:\n{translation}")
    return translation


#####################################
# 6. Demo 流程（供命令行测试使用，可在 Web 调用中不使用）
#####################################
def translation_workflow(user_input: str, max_rounds: int = 2):
    """
    整体流程：
      1. 生成初始提示文本（instruction）；
      2. 使用用户编辑后的提示调用本地 Qwen 模型生成 prompt0；
      3. 调用 DeepSeek API 获取初步翻译；
      4. Boss 模型审核译文，若反馈中建议修改，则迭代重新生成提示后调用 DS；
      5. 返回最终译文。
    """
    write_log(f"==== 开始交互流程，用户输入需求 ====\n{user_input}")
    print(f"\n=== 用户输入需求: {user_input} ===\n")

    # 生成初始提示文本供用户编辑
    initial_instruction = generate_instruction_text(user_input)
    # 此处用户可编辑后提交给本地 Qwen 模型
    prompt0 = call_local_qwen_with_instruction(initial_instruction)
    print(f"[Boss 模型] 生成的 Prompt0:\n{prompt0}\n")

    ds_translation = call_deepseek_api(prompt0)
    print(f"[DeepSeek 大模型] 返回的翻译:\n{ds_translation}\n")

    review_feedback = review_translation_with_boss(prompt0, ds_translation)
    print(f"[Boss 模型] 审查反馈:\n{review_feedback}\n")

    # 示例：若反馈中包含修改建议，则使用反馈生成修订后的 prompt 重新调用 DS
    if any(keyword in review_feedback for keyword in ["改写", "需修改", "建议"]):
        revised_prompt = "请根据以下反馈重新生成翻译提示：" + review_feedback
        write_log("发现修改建议，重新生成提示并调用 DS。")
        ds_translation = call_deepseek_api(revised_prompt)
        print(f"[DeepSeek 大模型] 修订后返回的翻译:\n{ds_translation}\n")
    write_log("==== 交互流程结束 ====")
    return ds_translation

def get_local_model():
    global model
    return model

def get_local_tokenizer():
    global tokenizer
    return tokenizer
    
if __name__ == "__main__":
    # 简单命令行示例（供调试使用）
    user_query = (
        "请把这首中国古诗翻译成符合新西兰英语习惯的诗意文本：\n"
        "《静夜思》 - 李白\n"
        "床前明月光，疑是地上霜。举头望明月，低头思故乡。"
    )
    final_result = translation_workflow(user_query, max_rounds=3)
    print("=== 最终译文 ===")
    print(final_result)
    log_file.close()
