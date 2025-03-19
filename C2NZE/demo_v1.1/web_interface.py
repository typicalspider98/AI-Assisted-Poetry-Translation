import gradio as gr
import translation_logic as translation_logic
import tkinter as tk
from tkinter import filedialog
# import translation_logic_todo as translation_logic

# 新增：根据用户输入诗句生成初始提示文本（instruction）
def get_initial_instruction(user_input: str):
    """
    根据用户输入诗句生成初始提示文本，供用户查看和编辑
    """
    instruction_text = translation_logic.generate_instruction_text(user_input)
    return instruction_text


def call_qwen_with_instruction(instruction_text: str):
    """
    将用户编辑后的提示文本提交给本地 Qwen 模型生成最终的 prompt0
    """
    prompt0 = translation_logic.call_local_qwen_with_instruction(instruction_text)
    return prompt0


def call_ds_translation_web(prompt0: str):
    """
    将 prompt0 提交给 DeepSeek API 获取翻译（translation1）
    """
    translation1 = translation_logic.call_deepseek_api(prompt0)
    return translation1


def review_translation_web(prompt0: str, ds_translation: str):
    """
    将 DS 返回的翻译提交给 Qwen 审查，得到审核反馈
    """
    review_feedback = translation_logic.review_translation_with_boss(prompt0, ds_translation)
    return review_feedback


def call_ds_revision_web(review_feedback: str):
    """
    根据用户编辑后的审核意见生成新的提示，并调用 DS 获取翻译（translation2）
    """
    revised_prompt = "请根据以下反馈重新生成翻译提示：" + review_feedback
    translation2 = translation_logic.call_deepseek_api(revised_prompt)
    return translation2

def select_model_folder():
    """ 打开文件夹选择对话框，返回用户选中的路径 """
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    folder_selected = filedialog.askdirectory()  # 选择文件夹
    return folder_selected if folder_selected else "未选择文件夹"

# 构建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("## 中文古诗翻译（Qwen14B + DeepSeek 多轮交互示例）\n"
                "注意：所有交互日志将保存在 logs 文件夹中。")

    # 第一行：输入诗歌（预设模板）
    with gr.Row():
        input_poetry = gr.Textbox(
            label="诗歌输入",
            lines=6,
            value="请输入中国古诗，例如：\n《静夜思》\n床前明月光，疑是地上霜。举头望明月，低头思故乡。"
        )
        btn_get_instruction = gr.Button("生成初始提示文本")

    # 第二行：展示初始提示文本（可编辑），用户可修改后提交给本地 Qwen 模型
    with gr.Row():
        textbox_instruction = gr.Textbox(label="初始提示文本（可编辑）", lines=8)
        model_path_display = gr.Textbox(label="当前模型路径", interactive=False)
        btn_select_model = gr.Button("选择本地模型文件夹")
        model_token_input = gr.Textbox(label="本地模型 Token", placeholder="输入 Token")
        btn_set_model_token = gr.Button("设置本地模型 Token")
        btn_submit_instruction = gr.Button("提交提示文本给 Boss")


    # 第三行：展示 Qwen14B 生成的 Prompt0
    with gr.Row():
        textbox_prompt0 = gr.Textbox(label="Boss生成的 Prompt0", lines=8)

    # 第四行：使用 Prompt0 调用 DeepSeek API 获取 Translation1
    with gr.Row():
        textbox_translation1 = gr.Textbox(label="DeepSeek 返回的 Translation1", lines=8)
        btn_submit_prompt = gr.Button("提交 Prompt0 给 DeepSeek")

    # 第五行：将 Translation1 提交给 Qwen 审查，得到审查反馈（可编辑）
    with gr.Row():
        textbox_review = gr.Textbox(label="Qwen14B 审查意见（可编辑）", lines=8)
        btn_call_ds_review = gr.Button("提交翻译审查")

    # 第六行：使用用户编辑后的审查反馈调用 DS 获取 Translation2
    with gr.Row():
        textbox_translation2 = gr.Textbox(label="DeepSeek 返回的 Translation2", lines=8)
        btn_submit_revision = gr.Button("修订后提交给 DS")

    # 如用户对 Translation2 不满意，可再次提交审查（循环）
    with gr.Row():
        btn_loop_review = gr.Button("再次提交审查（循环）")

    # 定义各按钮的交互逻辑

    btn_get_instruction.click(
        fn=get_initial_instruction,
        inputs=input_poetry,
        outputs=textbox_instruction
    )

    # 绑定按钮点击事件：选择模型文件夹
    btn_select_model.click(
    fn=select_model_folder,
    inputs=[],
    outputs=model_path_display
    )

    btn_set_model_token.click(
        fn=translation_logic.set_local_model_token,
        inputs=model_token_input,
        outputs=[]
    )

    btn_submit_instruction.click(
        fn=translation_logic.set_model_path,
        # fn=call_qwen_with_instruction,
        inputs=textbox_instruction,
        outputs=textbox_prompt0
    )

    btn_submit_prompt.click(
        fn=call_ds_translation_web,
        inputs=textbox_prompt0,
        outputs=textbox_translation1
    )

    btn_call_ds_review.click(
        fn=review_translation_web,
        inputs=[textbox_prompt0, textbox_translation1],
        outputs=textbox_review
    )

    btn_submit_revision.click(
        fn=call_ds_revision_web,
        inputs=textbox_review,
        outputs=textbox_translation2
    )

    btn_loop_review.click(
        fn=review_translation_web,
        inputs=[textbox_prompt0, textbox_translation2],
        outputs=textbox_review
    )

    demo.launch()
'''
if __name__ == "__main__":
    demo.launch()
'''
