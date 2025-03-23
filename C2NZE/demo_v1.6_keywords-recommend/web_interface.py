import gradio as gr
import translation_logic
import semantic_helper
import json

try:
    import tkinter as tk
    from tkinter import filedialog
    USE_TKINTER = True
except ImportError:
    USE_TKINTER = False

from gradio_checkboxgroupmarkdown import CheckboxGroupMarkdown

def select_model_folder():
    if USE_TKINTER:
        root = tk.Tk()
        root.withdraw()
        folder_selected = filedialog.askdirectory()
        return folder_selected if folder_selected else "未选择文件夹 | No folder selected"
    return "请手动输入模型路径 | Please enter the model path manually"

with gr.Blocks() as demo:
    gr.Markdown("## 中文古诗翻译（Qwen + DeepSeek 多轮交互示例）")

    # ==== 输入区域 ====
    with gr.Row():
        input_poetry = gr.Textbox(label="诗歌输入", lines=6, value="《静夜思》\n床前明月光，疑是地上霜。举头望明月，低头思故乡。")
        btn_get_instruction = gr.Button("生成初始提示文本")
        textbox_instruction = gr.Textbox(label="翻译提示文本（可编辑）", lines=8)

    with gr.Row():
        model_path_display = gr.Textbox(label="当前模型路径", interactive=True, value="/your/model/path")
        btn_select_model = gr.Button("选择模型文件夹")
        btn_set_model_path = gr.Button("确认模型路径")
        model_path_status = gr.Textbox(label="模型路径状态", interactive=False)

    with gr.Row():
        model_token_input = gr.Textbox(label="模型Token上限")
        btn_set_model_token = gr.Button("设置Token")
        model_token_status = gr.Textbox(label="Token设置结果", interactive=False)

    with gr.Row():
        btn_submit_boss_model = gr.Button("提交到本地Boss模型")
        textbox_prompt0 = gr.Textbox(label="Boss生成的Prompt0", lines=8)

    # ==== 关键词区域 ====
    with gr.Row():
        keyword_token_limit = gr.Textbox(label="关键词提示 Token 上限", value="128")
        btn_gen_prompt_keywords = gr.Button("生成关键词提取提示词")
        textbox_prompt_keywords = gr.Textbox(label="关键词提示词", lines=6)

    with gr.Row():
        btn_get_keywords = gr.Button("提取关键词")
        textbox_keywords_json = gr.Textbox(label="关键词 JSON", lines=4)

    # ==== 查询 & 分组展示区 ====
    btn_query_redis = gr.Button("查询向量数据库（TopK）")
    all_related_data = gr.State([])

    # 最多展示 10 个关键词的相关词推荐
    checkbox_groups = [
        CheckboxGroupMarkdown(choices=[], label=f"关键词{i+1} 的相关词", visible=False)
        for i in range(10)
    ]

    with gr.Column() as grouped_checkboxes_display:
        for cb in checkbox_groups:
            pass  # cb.render()

    # ==== 勾选 & 注入 ====
    btn_confirm_selection = gr.Button("确认选择关键词与相关词")
    textbox_selected_summary = gr.Textbox(label="最终选择结果（含说明和例句）", lines=12)

    btn_inject_keywords = gr.Button("注入关键词")
    textbox_final_prompt = gr.Textbox(label="注入后的提示词", lines=8)

    # ==== 翻译 + 审查 ====
    textbox_translation1 = gr.Textbox(label="DeepSeek返回的Translation1", lines=8)
    btn_submit_prompt = gr.Button("提交 Prompt0 给 DeepSeek")

    textbox_review = gr.Textbox(label="Qwen审查意见（可编辑）", lines=8)
    btn_call_ds_review = gr.Button("提交翻译审查")

    textbox_translation2 = gr.Textbox(label="DeepSeek返回的Translation2", lines=8)
    btn_submit_revision = gr.Button("修订后提交给 DS")

    btn_loop_review = gr.Button("再次提交审查（循环）")

    # ==== 功能绑定 ====
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

    # 查询相关词并更新分组勾选框
    btn_query_redis.click(
        fn=semantic_helper.query_related_terms_from_redis,
        inputs=textbox_keywords_json,
        outputs=all_related_data
    ).then(
        fn=semantic_helper.render_checkbox_groups_by_keyword,
        inputs=all_related_data,
        outputs=checkbox_groups
    )

    # 收集用户勾选项并展示为结构化 JSON
    btn_confirm_selection.click(
        fn=semantic_helper.collect_grouped_markdown_selection,
        inputs=checkbox_groups + [all_related_data],
        outputs=textbox_selected_summary
    )

    btn_inject_keywords.click(
        fn=semantic_helper.inject_keywords_into_prompt,
        inputs=[textbox_prompt0, textbox_selected_summary],
        outputs=textbox_final_prompt
    )

    demo.launch(share=True)
