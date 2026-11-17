---
name: retriever
description: 用自然语言检索并下载论文、GitHub 仓库/代码、数据集、网页到本地归档，并对已下载文档做本地问答。当用户想"找论文/下论文""找开源项目/下代码""搜数据集""把网页存下来"或问"已下载的文档里说了什么"时使用。通过 retrieve CLI 执行。
---

# Retriever Skill

`retrieve` 是本项目 CLI（仓库 `lxxxc147/retriever`）：输入一句自然语言，
自动完成「理解需求 → 选数据源 → 搜索 → 去重排序 → 下载 → manifest/历史」。

## 何时使用

- 找论文并下载 PDF（arXiv / Google Scholar）
- 找 GitHub 仓库、按内容搜代码文件、下载仓库 ZIP / Release / 单文件
- 找数据集（Hugging Face Hub）
- 通用网页搜索并保存正文为 Markdown
- 对**已经下载并解析过**的文档提问（本地问答，不走网络）

不要用本 Skill 做：实时股价/新闻查询（这是搜索下载工具，不是问答搜索引擎）。

## 环境前提

- `retrieve` 命令可用：源码目录 `uv run retrieve`，或
  `uv tool install git+https://github.com/lxxxc147/retriever.git`
- 可选 `.env`（不配也能用）：`GITHUB_TOKEN` 提高 GitHub 限流额度；
  `LLM_API_KEY` 启用 LLM 意图解析/摘要/问答（不配置自动降级为规则版）

## 命令速查

```bash
retrieve "找 2024 年 LLM agent 的综述" --limit 5     # 检索（意图自动路由数据源）
retrieve "transformer survey" --source arxiv          # 指定数据源
retrieve "..." --download                             # 检索 + 下载到 ./downloads/<task_id>/
retrieve "..." --download --parse                     # 再解析 PDF → ./parsed/
retrieve --ask "what problem does the attention mechanism solve"   # 本地问答（可加 --task <id>）
retrieve history                                      # 回看历史（含 task_id、manifest 路径）
retrieve                                              # 交互模式：refine / d N 下载 / N 看详情 / q
```

## 工作流建议

1. 先检索看结果（不下载），满意后加 `--download --parse`；
2. 论文类查询默认走 arXiv（无需 Key）；GitHub 匿名限流低，量大时配 Token；
3. 下载后想追问内容 → `--ask`，中文问题问英文论文时改用论文里的英文术语；
4. 每条下载都有 `manifest.json`（url/本地路径/sha256），引用溯源从这里取。

## 故障与限制

- 单个源超时/被限流不影响其他源，WARNING 里会给出分类原因（限流/鉴权/网络）；
- `web`（DuckDuckGo）、`scholar`、`huggingface` 在部分网络环境不可达，属环境问题；
- 扫描版 PDF 无文本层，只标记 `parse_quality: scan`；
- 本地问答是词汇级检索，不做向量召回；文档集很大时需要按 `--task` 缩小范围。
