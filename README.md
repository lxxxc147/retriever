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
uv run retrieve --help    # 查看 CLI（当前为 W1 骨架，Phase 1 末接入完整命令）
uv run pytest             # 跑测试
uv run ruff check .       # lint
```

## 目录结构

```
retrieval-agent/
├── pyproject.toml
├── .env.example            # 配置模板（含注释说明）
├── config/
│   ├── settings.yaml       # 限流、权重、路径等配置
│   └── sources.yaml        # 数据源注册表
├── src/
│   └── retriever/
│       ├── __init__.py
│       ├── cli.py          # CLI 入口（typer）
│       ├── models.py       # SearchQuery / SearchResult / Manifest 数据模型
│       ├── config.py       # 配置加载
│       ├── intent.py       # M1 需求解析
│       ├── router.py       # M2 路由调度
│       ├── pipeline.py     # M4 结果处理流水线
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
