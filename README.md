# Retriever — 智能检索与下载 Agent

用一句自然语言描述需求，自动完成「理解需求 → 选择数据源 → 搜索 → 筛选排序 → 下载 → 整理输出」全流程：检索论文 / GitHub 仓库 / 数据集 / 网页，并把文件规范地下载到本地归档。

**需求文档（Single Source of Truth）见 [PRD.md](PRD.md)。**

## 安装

依赖管理使用 [uv](https://docs.astral.sh/uv/)（会自动下载管理 Python）：

```bash
uv sync                 # 安装全部依赖（含 dev 组）
cp .env.example .env    # 填入自己的 GitHub Token（见文件内注释）
```

### 代理（可选，国内建议配置）

直连 `raw.githubusercontent.com` / `codeload.github.com` 经常在传输中途卡死，表现为
**搜索正常但下载失败**。在 `.env` 里配置即可全程走代理：

```bash
RETRIEVER_PROXY=http://127.0.0.1:7897   # 本机 Clash 混合端口
```

优先级：`RETRIEVER_PROXY`（`.env`）> `network.proxy`（`config/settings.yaml`）>
环境变量 `HTTP_PROXY` / `HTTPS_PROXY`；都为空则直连。

## 快速开始

```bash
uv run retrieve --help    # 查看 CLI
uv run pytest             # 跑测试
uv run ruff check .       # lint
```

## 下载与历史记录

**下载**统一走 `DownloadManager`（M5，PRD §2.5）：并发下载（默认 3 路）、按内容 sha256 去重、失败退避重试、失败明细写 `download_failures.log`、任务结束生成 `manifest.json`。落盘结构：

```
downloads/<task_id>/
├── arxiv/01_Attention_Is_All_You_Need.pdf
├── github/02_org_repo.zip
├── download_failures.log      # 只在有失败时出现
└── manifest.json              # task_id / query / 每条文件的 url、本地路径、大小、sha256
```

**历史记录**（X-4，PRD §2.7 F7.4）：每次 `retrieve` 结束都会往 `./history/<日期>.jsonl` 追加一条记录（查询、意图、解析器、数据源、结果数、下载成功/失败数、manifest 路径）。

```bash
$ retrieve history
Recent 1 task(s):
  2026-10-20 05:59  b82fe04f95c7  [paper]  transformer survey
      1 result(s), 1 downloaded, parser=llm
      manifest: downloads/b82fe04f95c7/manifest.json
```

- `retrieve` 不带参数进入交互模式（见下节）；`retrieve history` 专门用来回看历史；
- `--history-dir` 可改历史目录（默认 `./history`，已在 `.gitignore`）；
- 写历史失败只打 warning，不影响检索本身。

## 交互模式（F8.2 / F7.3）

不带参数运行 `retrieve`（或加 `-i/--interactive`）进入交互会话：首轮输入检索内容后，可以继续用自然语言收窄结果，并直接选择下载——不用再拼一整条命令：

```text
$ retrieve
Retriever 交互模式 — 输入检索内容开始，输入 help 查看命令，q 退出。
retrieve> transformer survey
Plan: intent=paper | keywords=transformer | parser=rules
Found 5 result(s):
...
— 继续输入可 refine 结果；d 下载全部、d N 下载第 N 条、N 看详情、help 帮助、q 退出
retrieve> 只要 2024 年的
Plan: intent=paper | keywords=transformer | parser=rules | years=2024-2024
Found 2 result(s):
...
retrieve> d 2
Downloading 1 file(s) in parallel -> downloads/<task_id> ...
retrieve> q
```

- 自然语言 refine 先对当前结果做本地过滤（年份、stars、语言、作者、分类、关键词）；过滤后为空时自动把条件合并进检索式重新搜索，不会悄悄返回空列表；
- `d N` 下载第 N 条（编号以当前列表为准）、`N` 查看详情；
- 每一轮 refine 和每一次下载都会记一条历史。

## PDF 解析（M6）

下载完成后可以把 PDF 解析成结构化 JSON（产物在 `parsed/<日期>/<task_id>/<文件名>.json`，便于溯源与后续问答）：

```bash
retrieve "transformer survey" --source arxiv --limit 1 --download --parse
# Parsed 1 PDF file(s) -> parsed (0 failed).
```

输出结构：

```json
{
  "meta": {
    "title": "A Survey on ...",
    "page_count": 11,
    "word_count": 3400,
    "sha256": "d5672fa5...",
    "parse_quality": "normal",
    "file_name": "01_A_Survey_on.pdf"
  },
  "content": {"body": "正文……", "references": ["[1] Zhang S. ...", "[2] Li M. ..."]}
}
```

- `parse_quality` 为 `normal` 或 `scan`：扫描版没有文本层，只标记不报错（PRD §9 风险表）；
- 标题优先级：PDF 元数据 > 页面最大字号文本 > 文件名；
- 解析是**可选能力**：`pymupdf` 无法加载时 `--parse` 只提示并跳过，不影响检索与下载；
- 只解析 `.pdf` 产物（GitHub 的 ZIP、网页的 Markdown 会跳过）。

## LLM 意图解析（Phase 2）

意图解析有两套实现，输出同一个 `ParsedIntent` 契约，对下游 Router 完全透明：

| 解析器 | 实现 | 成本 | 触发方式 |
|---|---|---|---|
| 规则版 | `intent.py`（正则 + 词典，PRD X-1） | 零成本、可复现 | `--parser rule`，或没配 Key 时的 `auto` |
| LLM 版 | `intent_llm.py` + `llm.py`（OpenAI 兼容 `/chat/completions`） | 一次模型调用 | `--parser llm`，或配了 Key 时的 `auto` |

```bash
retrieve "帮我找 2023 年以后 RAG 相关的 github 项目，stars 超过 500"   # auto：有 Key 走 LLM
retrieve "transformer survey" --parser rule                        # 强制规则版（离线、零成本）
retrieve "llm agent 综述" --parser llm                             # 强制 LLM 版
```

解析结果会在结果列表前打印一行 `Plan: ...parser=llm(deepseek-chat)`，方便确认这一次到底走了哪条路径。

**配置**（`.env`，不进版本库）：

```bash
LLM_API_KEY=sk-...                                  # 必填；也接受 DEEPSEEK_API_KEY
LLM_BASE_URL=https://api.deepseek.com/v1            # 可选，覆盖 settings.yaml 的 llm.base_url
LLM_MODEL=deepseek-chat                             # 可选，覆盖 llm.model
```

端点与模型默认值在 `config/settings.yaml` 的 `llm:` 段，任何 OpenAI 兼容端点（DeepSeek / Moonshot / 自建网关）都能直接换上。

**降级策略**（PRD §3 NF2，单点故障不影响主流程）：

| 情况 | 行为 |
|---|---|
| 没配 Key | `auto` 直接用规则版，不报警告 |
| 模型超时 / HTTP 错误 / Key 无效 | 规则版兜底，stderr 打一行 `warning:`，检索照常跑完 |
| 返回非法 JSON、意图取值越界、关键词为空 | 同上，规则版兜底 |
| 查询本身是 arXiv/GitHub URL | `auto` 不调用模型（URL 已精确，省一次 token） |

**测试**：LLM 相关单测全部用 `respx` mock（`tests/test_llm.py`）或注入假客户端（`tests/test_intent_llm.py`、`tests/test_cli.py`），`tests/conftest.py` 里还有一个 autouse fixture 把默认 LLM 客户端替换成未配置桩，保证本地 `.env` 里有 Key 时跑测试也不会真的打网络。

## 结果处理（M4 Pipeline）

检索结果在展示前统一过一遍 `pipeline.run_pipeline()`（PRD §2.4）：

| 步骤 | 做什么 | 配置 |
|---|---|---|
| 去重（F4.1） | 按 arXiv ID / DOI / GitHub `owner/repo` / 归一化 URL 判定同一结果，跨源重复只留一条，缺失的摘要/作者/日期/元数据从重复项补齐 | 无需配置 |
| 排序（F4.2） | 加权求和：来源可信度 + 时效性（按年指数衰减）+ 热度（引用数/stars 取对数归一） | `config/settings.yaml` 的 `ranking.weights` |
| 摘要（F4.3） | 每条结果一句话中文摘要 | `--summary` 开启 |

```bash
retrieve "RAG 数据集" --limit 3 --summary
```

```
Plan: intent=dataset | keywords=RAG | parser=llm(deepseek-flash)
Found 6 result(s):

[1] [GitHub] langgenius/dify
    URL: https://github.com/langgenius/dify
    Summary: Dify 是构建 Agent 工作流与 RAG 流水线的协作平台，支持云端或自托管部署。
```

摘要默认**一次批量调用**（不是每条一次），没有 Key 或调用失败时自动降级为"摘要首句 + 截断"的抽取式摘要，并在 stderr 提示一次。

排序权重可现场调，改完不用改代码：

```yaml
ranking:
  weights:
    source: 0.4        # 来源可信度：arxiv 1.0 / scholar 0.95 / github 0.9 / pdf 0.7 / web 0.6
    recency: 0.3       # 时效性：5 年半衰期
    popularity: 0.3    # 引用数或 stars：log1p(value)/log1p(1000)，封顶 1.0
```

## 目录结构

```
retrieval-agent/
├── pyproject.toml
├── .env.example            # 配置模板（含注释说明）
├── config/
│   ├── settings.yaml       # 限流、权重、路径、LLM 端点等配置
│   └── sources.yaml        # 数据源注册表
├── src/
│   └── retriever/
│       ├── __init__.py
│       ├── cli.py          # CLI 入口（argparse）
│       ├── models.py       # SearchQuery / SearchResult / Manifest 数据模型
│       ├── config.py       # 配置加载
│       ├── intent.py       # M1 需求解析（规则版）
│       ├── intent_llm.py   # M1 需求解析（LLM 版 + 规则兜底）
│       ├── llm.py          # LLM 客户端（OpenAI 兼容，走 HttpClient）
│       ├── router.py       # M2 路由调度
│       ├── pipeline.py     # M4 结果处理流水线（去重/加权排序/摘要装配）
│       ├── summarizer.py   # M4 一句话摘要（LLM 批量 + 抽取式兜底）
│       ├── downloader.py   # M5 下载管理
│       ├── parser/         # M6 内容解析
│       │   ├── pdf.py
│       │   └── code.py
│       ├── presenter.py    # M7 输出与交互
│       ├── history.py      # 历史记录
│       └── adapters/       # M3 数据源适配器
│           ├── base.py     # SearchAdapter 抽象基类
│           ├── example.py  # 示范 Adapter（可照抄）
│           ├── arxiv.py    # ← 成员A
│           ├── scholar.py  # ← 成员A（Phase 2）
│           ├── web.py      # ← 成员A（Phase 2）
│           └── github.py   # ← 成员B
├── tests/
│   ├── fixtures/           # mock 响应样本（JSON）
│   ├── adapters/
│   └── ...
├── downloads/              # 运行时生成（gitignore）
├── parsed/
├── history/
└── PRD.md                  # 需求文档
```

## 开发规范摘要（详见 PRD §7）

- **分支**：所有开发在 `feature/{模块}-{简要描述}` 分支进行；`main` 为保护分支，只接受 PR。
- **PR 就绪标准**：CI（lint + test）全绿、至少 1 个 approve、关联 Issue。
- **合并**：Squash Merge；禁止直接 push main、禁止 `git push -f`。
- **代码规范**：ruff（行宽 100）+ pre-commit；公开函数必须有 type hint 和 docstring；
  代码与注释用英文，PRD/Issue/Commit/Review 讨论用中文。
- **测试**：Adapter 单测用 `respx` mock HTTP，禁止打真实 API；CI 全绿才算 PR 就绪。
- **铁律**：
  1. Adapter 之间不允许互相 import，只能通过 `models.py` 传递数据。
  2. 任何网络请求必须经统一的 `HttpClient`（`http.py`），禁止裸调 HTTP。
