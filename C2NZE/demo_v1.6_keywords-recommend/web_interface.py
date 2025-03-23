import gradio as gr
import translation_logic
import semantic_helper  # 新增导入
from gradio_checkboxgroupmarkdown import CheckboxGroupMarkdown
import json

try:
    import tkinter as tk
    from tkinter import filedialog
    USE_TKINTER = True
except ImportError:
    USE_TKINTER = False


def select_model_folder():
    if USE_TKINTER:
        root = tk.Tk()
        root.withdraw()
        folder_selected = filedialog.askdirectory()
        return folder_selected if folder_selected else "未选择文件夹 | No folder selected"
    return "请手动输入模型路径 | Please enter the model path manually"


with gr.Blocks() as demo:
    gr.Markdown("## 中文古诗翻译（Qwen + DeepSeek 多轮交互示例）\n"
                "## Chinese Classical Poetry Translation (Qwen + DeepSeek Multi-round Interaction Example)\n"
                "注意：所有交互日志将保存在 logs 文件夹中。\nNote: All interaction logs will be saved in the logs folder.")

    # 输入诗歌 + 提示词生成
    with gr.Row():
        input_poetry = gr.Textbox(label="诗歌输入 Poetry input", lines=6, value="《静夜思》\n床前明月光，疑是地上霜。举头望明月，低头思故乡。")
        btn_get_instruction = gr.Button("生成初始提示文本 | Generate initial prompt text")
        textbox_instruction = gr.Textbox(label="翻译提示文本（可编辑）| Prompt text (editable)", lines=8)

    # 模型路径 + Token
    with gr.Row():
        # model_path_display = gr.Textbox(label="当前模型路径", interactive=True)
        model_path_display = gr.Textbox(label="当前模型路径 | Current model path", interactive=True, value="/workspace/Project-Code/AI-Assisted-Poetry-Translation/C2NZE/models/DeepSeek-R1-Distill-Qwen-1.5B")
        btn_select_model = gr.Button("选择本地模型文件夹")
        btn_set_model_path = gr.Button("确认模型路径")
        model_path_status = gr.Textbox(label="模型路径状态", interactive=False)

    with gr.Row():
        model_token_input = gr.Textbox(label="本地模型Token上限")
        btn_set_model_token = gr.Button("设置本地模型 Token")
        model_token_status = gr.Textbox(label="模型token状态", interactive=False)

    # 本地提示词生成
    with gr.Row():
        btn_submit_boss_model = gr.Button("提交到本地Boss模型")
        textbox_prompt0 = gr.Textbox(label="Boss生成的Prompt0", lines=8)

    # 设置关键词提示 Token 长度
    with gr.Row():
        keyword_token_limit = gr.Textbox(label="关键词提示 Token 上限", value="128")

    # 提取关键词提示词
    with gr.Row():
        btn_gen_prompt_keywords = gr.Button("生成关键词提取提示词")
        textbox_prompt_keywords = gr.Textbox(label="关键词提示词 Prompt", interactive=True, lines=6)

    # 调用 LLM 提取关键词
    with gr.Row():
        btn_get_keywords = gr.Button("使用本地模型提取关键词")
        textbox_keywords_json = gr.Textbox(label="关键词 JSON", interactive=True, lines=4)

    # 查询向量数据库 TopK
    with gr.Row():
        btn_query_redis = gr.Button("查询向量数据库（TopK）")
        checkbox_related_terms = CheckboxGroupMarkdown(choices=[], label="相关新西兰用词选择 | Select related NZ terms")

    # 注入关键词并生成新 Prompt
    with gr.Row():
        btn_inject_keywords = gr.Button("注入关键词")
        textbox_final_prompt = gr.Textbox(label="注入后的提示词", lines=8)

    # 翻译流程
    with gr.Row():
        textbox_translation1 = gr.Textbox(label="DeepSeek返回的Translation1", lines=8)
        btn_submit_prompt = gr.Button("提交 Prompt0 给 DeepSeek")

    with gr.Row():
        textbox_review = gr.Textbox(label="Qwen审查意见（可编辑）", lines=8)
        btn_call_ds_review = gr.Button("提交翻译审查")

    with gr.Row():
        textbox_translation2 = gr.Textbox(label="DeepSeek返回的Translation2", lines=8)
        btn_submit_revision = gr.Button("修订后提交给 DS")

    with gr.Row():
        btn_loop_review = gr.Button("再次提交审查（循环）")

    # 功能绑定
    btn_select_model.click(fn=select_model_folder, inputs=[], outputs=model_path_display)
    btn_set_model_path.click(fn=translation_logic.set_model_path, inputs=model_path_display, outputs=model_path_status)
    btn_set_model_token.click(fn=translation_logic.set_local_model_token, inputs=model_token_input, outputs=model_token_status)
    btn_get_instruction.click(fn=translation_logic.generate_instruction_text, inputs=input_poetry, outputs=textbox_instruction)
    btn_submit_boss_model.click(fn=translation_logic.call_local_qwen_with_instruction, inputs=[textbox_instruction, model_token_input], outputs=textbox_prompt0)
    btn_submit_prompt.click(fn=translation_logic.call_deepseek_api, inputs=textbox_prompt0, outputs=textbox_translation1)
    btn_call_ds_review.click(fn=translation_logic.review_translation_with_boss, inputs=[textbox_prompt0, textbox_translation1], outputs=textbox_review)
    btn_submit_revision.click(fn=translation_logic.call_deepseek_api, inputs=textbox_prompt0, outputs=textbox_translation2)
    btn_loop_review.click(fn=translation_logic.review_translation_with_boss, inputs=[textbox_prompt0, textbox_translation2], outputs=textbox_review)

    btn_gen_prompt_keywords.click(
        fn=lambda poem, limit: semantic_helper.build_keyword_prompt(poem, int(limit)),
        inputs=[input_poetry, keyword_token_limit],
        outputs=textbox_prompt_keywords
    )

    btn_get_keywords.click(
        fn=lambda prompt_text, limit: semantic_helper.extract_keywords_with_llm(prompt_text, int(limit)),
        inputs=[textbox_prompt_keywords, keyword_token_limit],
        outputs=textbox_keywords_json
    )


    btn_query_redis.click(
        fn=lambda json_text: gr.update(
            choices=semantic_helper.query_related_terms_from_redis(
                semantic_helper.display_keyword_options(json_text)
            )
        ),
        inputs=textbox_keywords_json,
        outputs=checkbox_related_terms
    )

    btn_inject_keywords.click(
        fn=semantic_helper.inject_keywords_into_prompt,
        inputs=[textbox_prompt0, checkbox_related_terms],
        outputs=textbox_final_prompt
    )

    demo.launch(share=True)
