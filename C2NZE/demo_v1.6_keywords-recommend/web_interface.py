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
    gr.Markdown("## 中文古诗翻译（Qwen + DeepSeek 多轮交互示例） | Chinese ancient poetry translation (Qwen + DeepSeek multi-round interaction example)")

    # ==== 输入区域 ====
    with gr.Row():
        with gr.Column(scale=2):
            input_poetry = gr.Textbox(label="诗歌输入 | Poetry Input", lines=6, value="《静夜思》\n床前明月光，疑是地上霜。举头望明月，低头思故乡。")
            btn_get_instruction = gr.Button("生成初始提示文本 | Generate initial prompt text")
            btn_get_instruction_EN = gr.Button("生成EN初始提示文本 | Generate initial english prompt text")
        with gr.Column(scale=6):
            textbox_instruction = gr.Textbox(label="翻译提示文本（可编辑） | Translation hint text (editable)", lines=8)

    with gr.Row():
        with gr.Column(scale=4):
            model_path_display = gr.Textbox(label="当前模型路径 | Current model path", interactive=True, value="/workspace/AI-Assisted-Poetry-Translation/C2NZE/models/DeepSeek-R1-Distill-Qwen-14B")
        with gr.Column(scale=2):
            btn_select_model = gr.Button("选择模型文件夹 | Select Model Folder")
            btn_set_model_path = gr.Button("确认模型路径 | Confirm the model path")
        with gr.Column(scale=4):
            model_path_status = gr.Textbox(label="模型路径状态 | Model Path Status", interactive=False)

    with gr.Row():
        with gr.Column(scale=1):
            model_token_input = gr.Textbox(label="模型Token上限 | The upper limit of the model token length", value="2048")
            btn_set_model_token = gr.Button("设置Token | Set Token")
            model_token_status = gr.Textbox(label="Token设置结果 | Token setting results", interactive=False)
            
        with gr.Column(scale=8):
            btn_submit_boss_model = gr.Button("提交到本地Boss模型 | Submit to local Boss model")
            textbox_prompt0 = gr.Textbox(label="Boss生成的Prompt0 | Boss spawned Prompt0", lines=10)

    # ==== 关键词区域 ====
    with gr.Row():
        with gr.Column(scale=2):
            btn_gen_prompt_keywords = gr.Button("生成关键词提取提示词 | Generate keyword extraction prompt words")
            btn_gen_prompt_keywords_EN = gr.Button("生成EN关键词提取提示词 | Generate keyword extraction english prompt words")
            keyword_token_limit = gr.Textbox(label="关键词提示 Token 上限 | The upper limit of the token length used for keyword generation", value="2048")
            btn_get_keywords = gr.Button("提取关键词 | Extract keywords")
        with gr.Column(scale=8):
            textbox_prompt_keywords = gr.Textbox(label="关键词提示词 | Keyword prompts", lines=8)

    with gr.Row():
        with gr.Column(scale=4):
            textbox_keywords_json = gr.Textbox(label="关键词 JSON | Keywords JSON", lines=4)
        with gr.Column(scale=2):
            btn_query_redis = gr.Button("查询向量数据库（TopK） | Query vector database (TopK)", scale=4)
            query_status = gr.Textbox(label="状态提示 | Status of the query", interactive=False, scale=1)

    # ==== 查询 & 分组展示区 ====
    # with gr.Row():
        # btn_query_redis = gr.Button("查询向量数据库（TopK） | Query vector database (TopK)", scale=4)
        # query_status = gr.Textbox(label="状态提示 | Status of the query", interactive=False, scale=1)
    # btn_query_redis = gr.Button("查询向量数据库（TopK）")
    # query_status = gr.Textbox(label="状态提示", interactive=False)
    all_related_data = gr.State([])

    # 最多展示 50 个关键词的相关词推荐（逐个定义并注册组件）
    checkbox_groups = []
    accordion_blocks = []

    grouped_checkboxes_display = gr.Column()

    for i in range(50):
        acc = gr.Accordion(f"关键词组 {i+1} | Keyword group {i+1}", open=False, visible=False)
        with acc:
            with gr.Row():
                cb_left = CheckboxGroupMarkdown(choices=[], label=f"关键词{i+1} 左侧 | Keyword {i+1} Left side", visible=True)
                cb_right = CheckboxGroupMarkdown(choices=[], label=f"关键词{i+1} 右侧 | Keyword {i+1} Right side", visible=True)

            btn_close_title = gr.Button("⬆️ 收起此关键词组（返回上方） | Collapse this keyword group (return to top)", variant="secondary")
            btn_close_title.click(fn=lambda: gr.update(open=False), inputs=[], outputs=acc)

            btn_next = gr.Button("🔽 查看下一个关键词组 | View next keyword group", variant="secondary")
            def make_next_fn(index):
                def inner():
                    updates = [gr.update(open=False) if i == index else gr.update() for i in range(50)]
                    if index + 1 < 50:
                        updates[index + 1] = gr.update(open=True)
                    return updates
                return inner
            btn_next.click(fn=make_next_fn(i), inputs=[], outputs=accordion_blocks)

        checkbox_groups.append([cb_left, cb_right])
        accordion_blocks.append(acc)
        # checkbox_groups.append(cb)
        # accordion_blocks.append(acc)

    # ==== 勾选 & 注入 ====
    btn_confirm_selection = gr.Button("确认选择关键词与相关词 | Confirm the selection of keywords and related words")
    textbox_selected_summary = gr.Textbox(label="最终选择结果（含说明和例句） | Final selection results (including instructions and examples)", lines=12)

    btn_inject_keywords = gr.Button("注入关键词 | Inject keywords")
    textbox_final_prompt = gr.Textbox(label="注入后的提示词 | Prompt words after injection", lines=8, interactive=True)

    # ==== 翻译 + 审查 ====
    btn_submit_prompt = gr.Button("提交 Prompt0 给 DeepSeek | Submit Prompt0 to DeepSeek")
    textbox_translation1 = gr.Textbox(label="DeepSeek返回的Translation1 | Translation1 returned by DeepSeek", lines=8, interactive=True)

    btn_call_ds_review = gr.Button("提交翻译审查 | Submit Translation Review")
    textbox_review = gr.Textbox(label="Qwen审查意见（可编辑） | Qwen review comments (editable)", lines=8, interactive=True)

    textbox_translation2 = gr.Textbox(label="DeepSeek返回的Translation2 | Translation2 returned by DeepSeek", lines=8)
    btn_submit_revision = gr.Button("修订后提交给 DS | Submit to DS after revision")

    btn_loop_review = gr.Button("再次提交审查（循环） | Submit for review again (loop)")

    # ==== 功能绑定 ====
    btn_select_model.click(fn=select_model_folder, inputs=[], outputs=model_path_display)
    btn_set_model_path.click(fn=translation_logic.set_model_path, inputs=model_path_display, outputs=model_path_status)
    btn_set_model_token.click(fn=translation_logic.set_local_model_token, inputs=model_token_input, outputs=model_token_status)
    btn_get_instruction.click(fn=translation_logic.generate_instruction_text, inputs=input_poetry, outputs=textbox_instruction)
    btn_get_instruction_EN.click(fn=translation_logic.generate_instruction_text_EN, inputs=input_poetry, outputs=textbox_instruction)
    btn_submit_boss_model.click(fn=translation_logic.call_local_qwen_with_instruction, inputs=[textbox_instruction, model_token_input], outputs=textbox_prompt0)

    btn_submit_prompt.click(fn=translation_logic.call_deepseek_api, inputs=textbox_final_prompt, outputs=textbox_translation1)

    btn_call_ds_review.click(fn=translation_logic.review_translation_with_boss, inputs=[textbox_final_prompt, textbox_translation1], outputs=textbox_review)
    btn_submit_revision.click(fn=translation_logic.call_deepseek_api, inputs=textbox_final_prompt, outputs=textbox_translation2)
    btn_loop_review.click(fn=translation_logic.review_translation_with_boss, inputs=[textbox_final_prompt, textbox_translation2], outputs=textbox_review)

    btn_gen_prompt_keywords.click(
        fn=lambda poem, limit: semantic_helper.build_keyword_prompt(poem, int(limit)),
        inputs=[input_poetry, keyword_token_limit],
        outputs=textbox_prompt_keywords
    )
    btn_gen_prompt_keywords_EN.click(
        fn=lambda poem, limit: semantic_helper.build_keyword_prompt_EN(poem, int(limit)),
        inputs=[input_poetry, keyword_token_limit],
        outputs=textbox_prompt_keywords
    )

    btn_get_keywords.click(
        fn=lambda prompt_text, limit: semantic_helper.extract_keywords_with_llm(prompt_text, int(limit)),
        inputs=[textbox_prompt_keywords, keyword_token_limit],
        outputs=textbox_keywords_json
    )

    btn_query_redis.click(
        fn=lambda: "正在查询向量数据库中，请稍候... | Querying the vector database, please wait...",
        inputs=[],
        outputs=query_status
    ).then(
        fn=semantic_helper.query_related_terms_from_redis,
        inputs=textbox_keywords_json,
        outputs=all_related_data
    ).then(
        fn=semantic_helper.render_checkbox_groups_by_keyword,
        inputs=all_related_data,
        outputs=[cb for pair in checkbox_groups for cb in pair]
    ).then(
        fn=lambda data: [gr.update(visible=i < len(data)) for i in range(50)],
        inputs=all_related_data,
        outputs=accordion_blocks
    ).then(
        fn=lambda: "✅ 查询完成！ | Query completed!",
        inputs=[],
        outputs=query_status
    )

    btn_confirm_selection.click(
        fn=semantic_helper.collect_grouped_markdown_selection,
        inputs=[cb for pair in checkbox_groups for cb in pair] + [all_related_data],
        outputs=textbox_selected_summary
    )

    btn_inject_keywords.click(
        fn=semantic_helper.inject_keywords_into_prompt,
        inputs=[textbox_prompt0, textbox_selected_summary],
        outputs=textbox_final_prompt
    )

    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
