import gradio as gr
import translation_logic

try:
    import tkinter as tk
    from tkinter import filedialog
    USE_TKINTER = True
except ImportError:
    USE_TKINTER = False  # 如果无法使用 tkinter，就禁用弹窗选择

def select_model_folder():
    """ 选择模型文件夹，返回选中的路径（如果支持 tkinter）"""
    if USE_TKINTER:
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        folder_selected = filedialog.askdirectory()  # 选择文件夹
        return folder_selected if folder_selected else "未选择文件夹"
    return "请手动输入模型路径"

# 构建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("## 中文古诗翻译（Qwen14B + DeepSeek 多轮交互示例）\n"
                "注意：所有交互日志将保存在 logs 文件夹中。")

    # 第一行：输入诗歌（预设模板）
    with gr.Row():
        input_poetry = gr.Textbox(
            label="诗歌输入,请输入中国古诗，例如：",
            lines=6,
            value="《静夜思》\n床前明月光，疑是地上霜。举头望明月，低头思故乡。"
        )
        btn_get_instruction = gr.Button("生成初始提示文本")

    # 第二行：用户选择模型路径 & 设定 Token
    with gr.Row():
        model_path_display = gr.Textbox(label="当前模型路径", interactive=True)
        btn_select_model = gr.Button("选择本地模型文件夹")  # 使用 tkinter 选择文件夹
        btn_set_model_path = gr.Button("确认模型路径")  # 确认最终路径
        model_path_status = gr.Textbox(label="模型路径状态", interactive=False)

    with gr.Row():
        model_token_input = gr.Textbox(label="本地模型 Token", placeholder="输入 Token")
        btn_set_model_token = gr.Button("设置本地模型 Token")
        model_token_status = gr.Textbox(label="模型token状态", interactive=False)


    # 第三行：展示 Qwen14B 生成的 Prompt0
    with gr.Row():
        textbox_instruction = gr.Textbox(label="翻译提示文本（可编辑）", lines=8)
        btn_submit_boss_model = gr.Button("提交本地 Boss 模型")
        textbox_prompt0 = gr.Textbox(label="Boss 生成的 Prompt0", lines=8)

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

    # 绑定按钮点击事件：选择模型文件夹
    btn_select_model.click(
        fn=select_model_folder,
        inputs=[],
        outputs=model_path_display
    )

    # 绑定按钮点击事件：确认并提交模型路径到 `translation_logic.py`
    btn_set_model_path.click(
        fn=translation_logic.set_model_path,
        inputs=model_path_display,
        outputs=model_path_status  # ✅ 确保用户看到路径已确认
    )

    # 绑定按钮点击事件：提交本地 Boss 模型（传入路径）
    btn_submit_boss_model.click(
        fn=translation_logic.call_local_qwen_with_instruction,
        inputs=[textbox_instruction, model_token_input],  # ✅ 确保传递模型路径
        outputs=textbox_prompt0
    )

    # 绑定按钮点击事件：设置本地模型 Token
    btn_set_model_token.click(
        fn=translation_logic.set_local_model_token,
        inputs=model_token_input,
        outputs=model_token_status  # ✅ 确保用户看到路径已确认
        # outputs=[]
    )

    btn_get_instruction.click(
        fn=translation_logic.generate_instruction_text,
        inputs=input_poetry,
        outputs=textbox_instruction
    )

    btn_submit_prompt.click(
        fn=translation_logic.call_deepseek_api,
        inputs=textbox_prompt0,
        outputs=textbox_translation1
    )

    btn_call_ds_review.click(
        fn=translation_logic.review_translation_with_boss,
        inputs=[textbox_prompt0, textbox_translation1],
        outputs=textbox_review
    )

    btn_submit_revision.click(
        fn=translation_logic.call_deepseek_api,
        inputs=textbox_prompt0,  # ✅ 确保是 Prompt0 进入 DeepSeek
        outputs=textbox_translation2
    )

    btn_loop_review.click(
        fn=translation_logic.review_translation_with_boss,
        inputs=[textbox_prompt0, textbox_translation2],
        outputs=textbox_review
    )

    # demo.launch(share=True)
    demo.launch()
