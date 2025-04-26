Semantic Helper Rerank Module
=============================

主要功能
--------
- 支持基于 Redis 向量数据库的关键词 TopK 检索
- 集成 FlagEmbedding + GTE reranker 进行双路打分，取最大得分作为排序依据
- 支持关键词相关释义与例句查询
- 提供前端交互界面（Accordion + Checkbox）渲染与用户选择
- 支持将用户选择注入翻译提示词（Prompt）中，用于翻译增强

适配工程
--------
- AI-Assisted-Poetry-Translation（诗歌人机翻译项目）

函数清单（Markdown版）
-----------------------

| 函数名 | 简要说明 |
|:---|:---|
| `get_embedding(text: str, model_id: int = 2)` | 加载指定嵌入模型，将文本转为均值池化向量 |
| `custom_sigmoid(x, temperature=1.5)` | 对 reranker 打分进行温度调节后的 Sigmoid 归一化 |
| `rerank_with_flag(pairs: List[List[str]])` | 使用 FlagEmbedding reranker 对句对打分 |
| `rerank_with_gte(pairs: List[List[str]])` | 使用 GTE reranker 对句对打分 |
| `search_topk_similar_batch(queries: List[str], top_k: int, model_id: int)` | （旧版，已停用）直接用余弦相似度在 Redis 中 TopK 检索 |
| `search_topk_with_reranker(queries: List[str], top_k: int, model_id: int)` | （新版✅）Redis检索后，用 Flag+GTE reranker双打分，取最大得分排序TopK |
| `query_related_terms_from_redis(json_text: str, top_k: int, model_id: int)` | 对关键词列表进行检索，查出相关词、释义和例句，供前端展示 |
| `render_checkbox_groups_by_keyword(all_data: list)` | 将查询到的关键词和相关词渲染为左右两列的 Gradio Checkbox 组件 |
| `collect_grouped_markdown_selection(*args)` | 收集用户在 Checkbox 中选择的相关词，并整理成统一 JSON |
| `update_accordion_labels(all_related_data)` | 动态更新 Accordion 折叠面板的标题 |
| `inject_keywords_into_prompt(prompt: str, selected_json: str)` | 将用户选择的关键词和释义注入到 Prompt 文本中，用于后续翻译提示增强 |
| `extract_base_word(redis_key: str)` | 从 Redis 的 key 名中提取基础词（去除 -数字编号后缀） |
| `display_keyword_options(json_text: str)` | 解析关键词 JSON，提取关键词列表 |
| `extract_keywords_with_llm(prompt_text: str, max_new_tokens: int = 128)` | 调用本地 LLM 提取诗歌关键词，返回 JSON 格式 |
| `build_keyword_prompt(poem_text: str, max_new_tokens: int = 128)` | 构建中文提示，指导模型提取英文关键词 |
| `build_keyword_prompt_EN(poem_text: str, max_new_tokens: int = 128)` | 构建英文提示，指导模型提取英文关键词 |
| `extract_keywords_regex(poem_text: str)` | 备用：用正则从诗歌文本中提取汉字短语作为关键词 |
| `if __name__ == "__main__"` | 防止直接运行本模块，提示应由主程序调用 |

模块调用关系（简图）
--------------------

```text
get_embedding
    ↓
search_topk_with_reranker
    ↓
query_related_terms_from_redis
    ↓
render_checkbox_groups_by_keyword
    ↓
collect_grouped_markdown_selection
    ↓
inject_keywords_into_prompt
```