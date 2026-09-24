# 检索类 Agent 产品需求文档（PRD）

| 项目名称 | Retriever — 智能检索与下载 Agent |
|---|---|
| 文档版本 | v1.0 |
| 文档状态 | 待评审 |
| 项目负责人 | （你，下称「负责人」） |
| 项目成员 | 新生 A、新生 B |
| 目标工期 | 4 个阶段，建议 6~8 周完成 MVP |

---

## 目录

1. [项目概述](#1-项目概述)
2. [功能需求](#2-功能需求)
3. [非功能需求](#3-非功能需求)
4. [技术架构](#4-技术架构)
5. [里程碑计划](#5-里程碑计划)
6. [团队分工](#6-团队分工)
7. [开发规范与协作流程](#7-开发规范与协作流程)
8. [验收标准](#8-验收标准)
9. [风险与应对](#9-风险与应对)
10. [附录](#10-附录)

---

## 1. 项目概述

### 1.1 项目背景

日常科研与开发工作中，我们经常需要：

- 找一篇论文（知道大概主题/作者，但不知道确切标题和链接）
- 找一个 GitHub 仓库或仓库里的某个具体文件
- 找一些数据集、技术文档

目前这些操作分散在多个平台，需要人工逐个搜索、筛选、下载、整理。我们希望做一个 **Agent**：用一句自然语言描述需求，它自动完成「理解需求 → 选择数据源 → 搜索 → 筛选排序 → 下载 → 整理输出」全流程。

### 1.2 项目目标

| 目标 | 说明 | 衡量指标 |
|---|---|---|
| G1 | 支持自然语言检索学术论文 | 输入主题/作者/年份，Top5 结果相关率 ≥ 70% |
| G2 | 支持自然语言检索 GitHub 仓库/文件 | 返回仓库可正常 clone/下载，文件路径正确 |
| G3 | 支持自动下载与本地归档 | 文件落盘成功率 ≥ 90%，命名规范可追溯 |
| G4 | 支持多轮交互 refine | 用户可对结果追加条件继续筛选 |
| G5 | 架构可扩展 | 新增一个数据源 Adapter ≤ 1 人日 |

### 1.3 非目标（本期不做）

- 不做搜索引擎级别的相关性排序算法（复用平台排序 + 简单加权）
- 不做付费墙的论文全文获取（只下载开放获取资源）
- 不做用户系统/多租户（本地单用户使用）
- 不做 Web UI（CLI 或 IM 对话形式交付）

### 1.4 术语表

| 术语 | 含义 |
|---|---|
| Adapter | 数据源适配器，封装某个平台（arXiv/GitHub…）的搜索与下载接口 |
| Slot | 槽位，从用户需求中提取的结构化字段（作者/年份/仓库名…） |
| Result | 统一的搜索结果结构（标题/摘要/链接/来源/元数据） |
| Pipeline | 结果处理流水线：去重 → 打分排序 → 摘要 |
| MVP | 最小可行产品，即 Phase 1 的交付物 |

---

## 2. 功能需求

> 优先级说明：**P0** = MVP 必须，**P1** = 第二阶段，**P2** = 增强项

### 2.0 功能总览

```
用户输入
   │
   ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ M1 需求解析  │ → │ M2 路由调度  │ → │ M3 搜索执行  │
└─────────────┘   └─────────────┘   └─────────────┘
                                           │
                    ┌──────────────────────┼──────────┐
                    ▼                      ▼          ▼
              ┌───────────┐         ┌──────────┐  ┌────────┐
              │ arXiv     │         │ GitHub   │  │ Web    │
              │ Adapter   │         │ Adapter  │  │ Adapter│
              └───────────┘         └──────────┘  └────────┘
                    │                      │          │
                    └──────────────────────┼──────────┘
                                           ▼
                                   ┌─────────────┐
                                   │ M4 结果处理  │  去重/排序/摘要
                                   └─────────────┘
                                           │
                                   ┌─────────────┐
                                   │ M5 下载管理  │  落盘/去重/重试
                                   └─────────────┘
                                           │
                                   ┌─────────────┐
                                   │ M6 内容解析  │  PDF/代码/网页
                                   └─────────────┘
                                           │
                                   ┌─────────────┐
                                   │ M7 交付输出  │  报告/追问/历史
                                   └─────────────┘
```

### 2.1 M1 需求解析模块（Intent Parser）

**职责**：把自然语言转成结构化检索任务。

| 编号 | 需求 | 优先级 |
|---|---|---|
| F1.1 | 意图分类：识别查询属于 论文/代码/数据集/网页/其他 | P0 |
| F1.2 | 论文槽位提取：关键词、作者、年份区间、arXiv ID、DOI | P0 |
| F1.3 | GitHub 槽位提取：仓库名、owner、语言、stars、文件路径、release 版本 | P0 |
| F1.4 | 精确查询识别：用户输入 URL 或 arXiv ID 时跳过搜索直接走下载 | P0 |
| F1.5 | 多轮澄清：关键槽位缺失时生成追问（如「您要找哪一年的论文？」） | P1 |
| F1.6 | 中英文混合查询处理 | P1 |

**输出契约**（SearchQuery）：

```json
{
  "intent": "paper | repo | code_file | dataset | web",
  "raw_query": "用户原始输入",
  "keywords": ["..."],
  "filters": {
    "authors": ["..."],
    "year_from": 2023,
    "year_to": 2025,
    "language": "python",
    "min_stars": 100
  },
  "limit": 5
}
```

### 2.2 M2 路由调度模块（Router）

**职责**：根据意图选择数据源及调用顺序。

| 编号 | 需求 | 优先级 |
|---|---|---|
| F2.1 | 按意图路由：paper→[arXiv, Scholar]；repo→[GitHub]；code_file→[GitHub Code Search] | P0 |
| F2.2 | 支持多源并行调用，聚合结果 | P0 |
| F2.3 | 路由配置外部化（JSON/YAML），不改代码可增删数据源 | P1 |
| F2.4 | 某数据源失败时自动降级到备选源 | P1 |

### 2.3 M3 搜索执行模块（Search Adapters）

**职责**：对接各平台 API，统一返回 Result 列表。所有 Adapter 实现统一接口（见 §4.4）。

| 编号 | 需求 | 优先级 | 负责 |
|---|---|---|---|
| F3.1 | arXiv Adapter：关键词/作者/分类搜索，支持分页 | P0 | 成员A |
| F3.2 | GitHub 仓库搜索 Adapter | P0 | 成员B |
| F3.3 | GitHub Code Search Adapter（按文件内容/路径搜索） | P0 | 成员B |
| F3.4 | GitHub 文件内容读取（raw.githubusercontent / MCP） | P0 | 成员B |
| F3.5 | Scholar Adapter（引用 Google Scholar） | P1 | 成员A |
| F3.6 | Web Adapter（通用搜索 + 网页正文提取） | P1 | 成员A |
| F3.7 | HuggingFace / Kaggle 数据集 Adapter | P2 | 待定 |
| F3.8 | 每个 Adapter 内置限流与重试（指数退避，最多 3 次） | P0 | 双方各自负责 |

**统一结果契约**（SearchResult）：

```json
{
  "source": "arxiv | github | scholar | web",
  "title": "...",
  "abstract": "...",        
  "url": "https://...",
  "download_url": "https://... (可空)",
  "published_at": "2024-03-01",
  "authors": ["..."],
  "extra": {
    "citations": 120,
    "stars": 2300,
    "language": "python",
    "license": "MIT"
  }
}
```

### 2.4 M4 结果处理模块（Result Pipeline）

**职责**：让结果「值得看」。

| 编号 | 需求 | 优先级 |
|---|---|---|
| F4.1 | 跨源去重（按规范化 URL / arXiv ID / DOI） | P0 |
| F4.2 | 打分排序：来源权重 + 时效性 + 引用数/stars 加权 | P0 |
| F4.3 | 每条结果生成一句话中文摘要（LLM 或规则截取） | P1 |
| F4.4 | 元数据补全（缺失字段尽力补全或标注 unknown） | P1 |

### 2.5 M5 下载管理模块（Download Manager）

**职责**：把文件安全、规范地保存到本地。

| 编号 | 需求 | 优先级 |
|---|---|---|
| F5.1 | 统一落盘路径：`./downloads/{日期}/{source}/` | P0 |
| F5.2 | 命名规范：`{序号}_{安全化标题}.{ext}`，冲突自动加序号 | P0 |
| F5.3 | 下载前按内容 hash 去重，已存在则跳过 | P0 |
| F5.4 | 支持单文件下载（PDF / 代码文件 / 网页快照） | P0 |
| F5.5 | 支持 GitHub 整仓 ZIP 下载与 Release 资产下载 | P1 |
| F5.6 | 失败重试 + 失败原因记录（写 `download_failures.log`） | P0 |
| F5.7 | 大文件（>100MB）下载进度提示 | P1 |
| F5.8 | 下载完成后生成清单文件 `manifest.json`（含元数据与本地路径） | P0 |

**manifest.json 示例**：

```json
{
  "task_id": "20260214-001",
  "query": "LLM agent survey 2024",
  "created_at": "2026-02-14T10:30:00",
  "files": [
    {
      "title": "A Survey on LLM-based Agents",
      "source": "arxiv",
      "url": "https://arxiv.org/abs/2401.xxxx",
      "local_path": "downloads/2026-02-14/arxiv/1_A_Survey_on_LLM-based_Agents.pdf",
      "size_bytes": 1234567,
      "sha256": "..."
    }
  ]
}
```

### 2.6 M6 内容解析模块（Content Parser）

**职责**：下载后的文件「可读、可检索」。

| 编号 | 需求 | 优先级 |
|---|---|---|
| F6.1 | PDF → 纯文本提取（标题/正文/参考文献分节） | P0 |
| F6.2 | 代码文件 → 识别语言、提取文件树、定位入口文件 | P1 |
| F6.3 | 网页 → 正文提取（去导航/广告），保存为 Markdown | P1 |
| F6.4 | 解析结果写入 `./parsed/{日期}/` 供后续问答使用 | P1 |

### 2.7 M7 交付输出模块（Presenter）

**职责**：面向用户的结果呈现与多轮交互。

| 编号 | 需求 | 优先级 |
|---|---|---|
| F7.1 | 结构化报告输出：序号/标题/一句话摘要/链接/本地路径/关键元数据 | P0 |
| F7.2 | 输出后提供操作提示：`[d] 下载全部  [d 2] 下载第2条  [m] 继续筛选` | P0 |
| F7.3 | 多轮 refine：基于上一轮结果追加条件再过滤 | P1 |
| F7.4 | 检索历史落盘 `./history/`，支持 `history` 命令回看 | P1 |
| F7.5 | 失败时给出可操作的原因说明（源站限流/需要登录/文件过大） | P0 |

**输出样式示例**：

```
🔍 查询：LLM agent survey 2024（论文）
✅ 找到 5 条结果（已去重，按相关性排序）

1. A Survey on LLM-based Autonomous Agents
   📄 一篇 2024 年的 LLM Agent 综述，涵盖规划、记忆、工具使用。
   🔗 https://arxiv.org/abs/2401.xxxx
   📊 引用 1200+ ｜ NeurIPS 2024
   💾 已下载 → downloads/2026-02-14/arxiv/1_A_Survey....pdf

[d] 下载全部  [d 3] 下载第3条  [s 作者=Wang] 继续筛选
```

### 2.8 CLI 入口

| 编号 | 需求 | 优先级 |
|---|---|---|
| F8.1 | `retrieve "查询语句"` 一次性检索 | P0 |
| F8.2 | `retrieve` 进入交互模式，支持多轮 | P0 |
| F8.3 | `retrieve --source arxiv "查询"` 指定数据源 | P1 |
| F8.4 | `retrieve history / config` 辅助命令 | P1 |

---

## 3. 非功能需求

| 编号 | 类别 | 需求 |
|---|---|---|
| NF1 | 性能 | 单源搜索响应 ≤ 10s（受源站 API 限制除外）；多源并行聚合 ≤ 15s |
| NF2 | 可靠性 | 任一 Adapter 崩溃不影响其他源；失败信息对用户可见 |
| NF3 | 可扩展 | 新增数据源只需：实现接口 + 注册配置，不改动框架代码 |
| NF4 | 可配置 | API Key、限流参数、下载路径、排序权重全部走配置文件 |
| NF5 | 日志 | 全链路日志：请求/响应/下载/失败，DEBUG 级可排查 |
| NF6 | 合规 | 尊重 robots 与 API 速率限制；只下载开放获取内容；遵守各平台 ToS |
| NF7 | 安全 | API Key 只存 `.env`，禁止写入代码与日志；不做凭证外发 |
| NF8 | 可测试 | 每个 Adapter 有 mock 单元测试；核心 Pipeline 有集成测试 |

---

## 4. 技术架构

### 4.1 技术选型

| 层级 | 选型 | 理由 |
|---|---|---|
| 语言 | Python 3.11+ | 生态最全（PDF/HTTP/数据），新手友好 |
| 并发 | asyncio + httpx | 多源并行搜索 |
| CLI | typer | 代码即文档，新手易上手 |
| 配置 | pydantic-settings + .env | 类型安全 + 环境隔离 |
| GitHub 能力 | GitHub REST API（优先）/ MCP 可选 | REST 直观易学，MCP 作为进阶 |
| PDF 解析 | pymupdf (fitz) | 快、纯文本提取质量高 |
| 日志 | loguru | 简单好看 |
| 测试 | pytest + pytest-asyncio + respx（mock HTTP） | 业界标准组合 |
| 包管理 | uv（或 pip + venv） | 快、lock 文件可复现 |
| 代码质量 | ruff +  pre-commit | 统一风格 |

### 4.2 目录结构

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
│           ├── arxiv.py    # ← 成员A
│           ├── scholar.py  # ← 成员A（Phase 2）
│           ├── web.py      # ← 成员A（Phase 2）
│           └── github.py   # ← 成员B（repo + code search + 文件）
├── tests/
│   ├── fixtures/           # mock 响应样本（JSON）
│   ├── test_intent.py
│   ├── test_router.py
│   ├── test_pipeline.py
│   ├── test_downloader.py
│   ├── adapters/
│   │   ├── test_arxiv.py   # ← 成员A
│   │   ├── test_github.py  # ← 成员B
│   │   └── ...
├── downloads/              # 运行时生成（gitignore）
├── parsed/
├── history/
└── docs/
    └── PRD.md              # 本文件
```

### 4.3 核心接口定义（骨架代码）

**所有 Adapter 必须实现：**

```python
# src/retriever/adapters/base.py
from abc import ABC, abstractmethod
from ..models import SearchQuery, SearchResult, DownloadResult
from pathlib import Path

class SearchAdapter(ABC):
    name: str                # 数据源标识，如 "arxiv"
    rate_limit_qps: float    # 每秒最大请求数

    @abstractmethod
    async def search(self, query: SearchQuery) -> list[SearchResult]: ...

    @abstractmethod
    async def download(self, result: SearchResult, dest_dir: Path) -> DownloadResult: ...
```

**下载管理器接口：**

```python
# src/retriever/downloader.py
class DownloadManager:
    async def download_all(self, results: list[SearchResult],
                           task_dir: Path) -> list[DownloadResult]:
        """并发下载（信号量限流），hash 去重，写 manifest.json"""
```

**项目唯一的两条铁律：**

1. Adapter 之间不允许互相 import，只能通过 `models.py` 里的数据结构传递数据。
2. 任何网络请求必须经统一的 `HttpClient`（自动限流 + 重试 + 日志），不允许在 Adapter 里裸调 `httpx.get`。

### 4.4 外部依赖清单（按模块）

| 模块 | 外部依赖 |
|---|---|
| M3 arXiv | arXiv HTTP API（`export.arxiv.org/api/query`，无需 Key） |
| M3 GitHub | GitHub REST API（`api.github.com`，search 接口需要 token，免费申请） |
| M3 Scholar | SerpAPI / scholar 库（有免费额度）或后续接 MCP |
| M3 Web | DuckDuckGo / Bing（免费层） + trafilatura（正文提取） |
| M5 下载 | httpx + aiofiles |

---

## 5. 里程碑计划

> 总计约 6~8 周（视新人投入时间而定）。每个 Phase 结束做一次 **Sprint Review + 复盘会**。

### Phase 1：骨架 + 双数据源 MVP（第 1~3 周）✅ 本期核心

| 周 | 交付物 | 验收 |
|---|---|---|
| W1 | 项目骨架：仓库、CI、配置、models、HttpClient、base adapter；成员各自跑通 hello-world adapter | 仓库可 `uv sync && pytest` 通过空骨架测试 |
| W2 | 成员A：arXiv 搜索 + 论文 PDF 下载；成员B：GitHub 仓库搜索 + 单文件下载 | 各自 Adapter 单测通过，能返回符合契约的 SearchResult |
| W3 | 整合：Intent → Router → 搜索 → 简单输出 → 下载落盘；CLI 可用；manifest.json | 输入「找 2024 年 LLM agent 综述」能出 5 条结果并下载 PDF |

**Phase 1 出口标准（= MVP）**：见 §8.1。

### Phase 2：质量与体验（第 4~5 周）

- Pipeline：去重、加权排序、一句话摘要
- GitHub：Code Search、整仓 ZIP、Release 下载
- Web Adapter 上线
- Scholar Adapter 上线（成员A）
- 多轮 refine、历史记录

### Phase 3：健壮性与内容解析（第 6 周）

- PDF 正文解析、代码文件树解析
- 失败降级链路（某源挂了自动切备选）
- 性能优化（并发、缓存）
- 全面的错误提示与日志

### Phase 4：扩展与打磨（第 7~8 周，按需）

- 数据集源（HuggingFace）
- manifest 驱动的本地知识库问答
- 打包发布 / 给 Kimi Work 接入 Skill 化

### 里程碑看板（建议直接抄到 GitHub Projects）

| Phase | 关键节点 | Owner |
|---|---|---|
| M1 骨架就绪 | W1 周五 | 负责人 |
| M2 双 Adapter 就绪 | W2 周五 | A / B |
| M3 MVP 发布 | W3 周五 | 负责人 |
| M4 质量升级 | W5 周五 | A / B |
| M5 健壮性 | W6 周五 | 全员 |

---

## 6. 团队分工

### 6.1 人员配置

| 角色 | 人员 | 职责 |
|---|---|---|
| 负责人 / Tech Lead | 你 | 架构设计、公共模块（models/config/http）、任务拆解、Code Review、集成与发布、排期与风险管理 |
| 成员 A | 新生 | **论文检索线**：arXiv → Scholar → Web Adapter |
| 成员 B | 新生 | **代码检索线**：GitHub Adapter（仓库/代码搜索/文件下载）+ 下载管理器 |

> 分工原则：**纵向切分数据源**，每人一条线从搜索到下载端到端负责——依赖少、成就感强、验收标准清晰。公共模块由负责人先行搭好，新人只做「填 Adapter」。

### 6.2 任务拆解与认领

#### 公共任务（负责人，W1 完成）

| 编号 | 任务 | 产出 | 预估 |
|---|---|---|---|
| T-0 | 建仓、分支保护、pre-commit、CI（lint + test） | 可提交的仓库骨架 | 0.5 天 |
| T-1 | `models.py` 全部数据模型 + 校验 | 契约代码 + docstring | 0.5 天 |
| T-2 | `config.py` + `settings.yaml` + `.env.example` | 配置加载 + 模板 | 0.5 天 |
| T-3 | `http.py` 统一 HttpClient（限流/重试/日志） | 公共客户端 | 1 天 |
| T-4 | `adapters/base.py` 抽象基类 + 一个 ExampleAdapter（示范） | 可照抄的模板 | 0.5 天 |
| T-5 | Sprint 看板 + 任务 Issue 录入 | GitHub Projects | 0.5 天 |

#### 成员 A 任务线（论文线）

| 编号 | 任务 | 对应需求 | Phase | 预估 | 产出物 |
|---|---|---|---|---|---|
| A-1 | arXiv Adapter：search（关键词/作者/分类/年份过滤、分页） | F3.1 | P1 | 2 天 | `arxiv.py` + 单测 |
| A-2 | arXiv Adapter：download（PDF 直链下载） | F3.1/F5.4 | P1 | 1 天 | download 实现 + 单测 |
| A-3 | arXiv 精确 ID/URL 直达（跳过搜索） | F1.4 | P1 | 0.5 天 | 路由协作 |
| A-4 | Scholar Adapter（引用数获取、排序字段） | F3.5 | P2 | 2 天 | `scholar.py` |
| A-5 | Web Adapter：通用搜索 + 正文提取 | F3.6 | P2 | 2 天 | `web.py` |
| A-6 | PDF 文本解析（与 B-6 二选一或协同） | F6.1 | P3 | 2 天 | `parser/pdf.py` |

#### 成员 B 任务线（代码线）

| 编号 | 任务 | 对应需求 | Phase | 预估 | 产出物 |
|---|---|---|---|---|---|
| B-1 | GitHub 仓库搜索 Adapter（关键词、语言、stars、更新时间过滤） | F3.2 | P1 | 2 天 | `github.py` + 单测 |
| B-2 | GitHub 单文件下载（raw 内容 + 正确命名） | F3.4/F5.4 | P1 | 1 天 | download 实现 |
| B-3 | GitHub Code Search Adapter（按路径/内容搜文件） | F3.3 | P1 | 1.5 天 | code search 实现 |
| B-4 | 整仓 ZIP 下载 + Release 资产下载 | F5.5 | P2 | 1.5 天 | 扩展 download |
| B-5 | DownloadManager：并发限流、hash 去重、manifest.json、失败日志 | F5.1~F5.8 | P1 | 2 天 | `downloader.py`（核心模块，负责人 Review 逐行过） |
| B-6 | 代码解析：文件树/入口识别（与 A-6 协同） | F6.2 | P3 | 1 天 | `parser/code.py` |

#### 交叉任务（按能力和兴趣认领）

| 编号 | 任务 | 建议 |
|---|---|---|
| X-1 | Intent Parser（规则版：关键词/正则/URL 识别） | P1 由负责人写，P2 谁手快谁重构 |
| X-2 | Result Pipeline（去重 + 加权排序） | P2 由负责人写骨架，A/B 补打分维度 |
| X-3 | CLI 交互与输出排版 | P1 负责人，P2 交叉 review |
| X-4 | 历史记录模块 | P2 空闲方认领 |

### 6.3 协作节奏

| 事项 | 频率 | 形式 | 参与 |
|---|---|---|---|
| Standup | 每天 15 分钟 | 文字版三问：昨天做了什么/今天做什么/有什么阻塞 | 全员 |
| Code Review | 每个 PR 必须 | GitHub PR Review，至少 1 人 approve 才能合 | 全员 |
| Sprint Review | 每 Phase 末 | 演示 + 复盘（做了什么/卡点/下次改进） | 全员 |
| 技术答疑 | 随时 | Issue 区提问，负责人 24h 内响应 | A/B 发起 |

### 6.4 新人成长路径设计（重要）

| 阶段 | 目标 | 给新人的支持 |
|---|---|---|
| 第 1 周 | 跑通仓库 + 照 ExampleAdapter 写出第一个 Adapter | 负责人提供模板、结对编程 2 次 |
| 第 2~3 周 | 独立完成线内任务，学会写单测 | PR 逐行 review，给参考文档 |
| 第 4~5 周 | 接触核心模块（DownloadManager / Pipeline） | 让新人主讲一次设计思路 |
| 第 6 周+ | 能独立认领 X 任务 / 带 Phase 4 | 轮换负责 Sprint Review 主持 |

---

## 7. 开发规范与协作流程

### 7.1 Git 工作流（GitHub Flow 简化版）

```
main ──────────────────────────────── (保护分支，只接受 PR)
  ↑      ↑        ↑
feature/arxiv-search   feature/github-adapter   feature/downloader
```

**规则：**
1. 所有开发在 `feature/{模块}-{简要描述}` 分支进行。
2. 提交信息格式：`feat(arxiv): add author filter` / `fix(github): handle rate limit 403`。
3. PR 必须：通过 CI（lint+test）、有至少 1 个 approve、关联 Issue。
4. 合并用 **Squash Merge**，保持 main 历史干净。
5. 禁止直接 push main；禁止 `git push -f` 共享分支。

### 7.2 开发前环境准备（新人第一天完成）

```bash
# 1. 克隆仓库
git clone <repo-url> && cd retrieval-agent

# 2. 安装 uv（如未安装）
pip install uv

# 3. 安装依赖
uv sync

# 4. 配置环境变量
cp .env.example .env   # 填入自己的 GitHub Token

# 5. 安装 pre-commit
pre-commit install

# 6. 验证
pytest                 # 应全部通过
ruff check .           # 应无报错
retrieve --help        # CLI 应可用
```

### 7.3 代码规范

- **风格**：ruff 默认规则集，行宽 100，pre-commit 自动修。
- **类型标注**：所有公开函数必须写 type hint（新人最容易忽视，Review 重点）。
- **文档**：模块/类/公开函数必须有 docstring，复杂逻辑行内注释。
- **禁止**：裸 `except:`、print 调试（用 loguru）、硬编码密钥/路径、在 Adapter 里裸调 HTTP。
- **语言**：代码与注释用英文；PRD、Issue、Commit、Review 讨论用中文。

### 7.4 测试规范

- 每个 Adapter 单测覆盖率 ≥ 80%，用 `respx` mock HTTP，**禁止测试打真实 API**。
- fixtures 目录存真实 API 响应样本（脱敏），测试断言解析正确性。
- 核心流水线（Pipeline / Downloader）必须有集成测试。
- CI 全绿才算 PR 就绪。

### 7.5 Issue / PR 模板

**Issue 模板要点**：任务编号（A-1/B-3）、对应 PRD 需求编号、验收标准、预估工时、截止 Phase。

**PR 模板要点**：

```markdown
## 关联 Issue
closes #12 (A-1)

## 改动说明
实现了 arXiv Adapter 的 search 接口，支持关键词/作者/年份过滤……

## 自测情况
- [x] 单测通过（新增 8 个用例）
- [x] 手动跑通：`retrieve "llm agent survey" --source arxiv`
- [x] manifest.json 输出正确

## 需要 Reviewer 重点看
分页逻辑第 xx 行的边界处理
```

---

## 8. 验收标准

### 8.1 MVP 验收（Phase 1 出口，全部满足才算过）

| # | 验收项 | 验证方式 |
|---|---|---|
| V1 | 输入「LLM agent 综述 2024」，arXiv 返回 ≥ 5 条含标题/摘要/链接的结果 | CLI 实际运行 |
| V2 | 输入「找 RAG 的 github 项目」，GitHub 返回 stars/语言/更新时间字段完整的结果 | CLI 实际运行 |
| V3 | 选择结果后可下载 PDF 到 `downloads/{日期}/arxiv/`，文件名规范无非法字符 | 检查磁盘 |
| V4 | 下载完成后生成 manifest.json，字段完整、路径真实存在 | 检查文件 |
| V5 | 断网/源站 403 时，程序不崩溃，给出可读的错误原因 | 模拟测试 |
| V6 | `pytest` 全绿，Adapter 单测覆盖率 ≥ 80% | CI 报告 |
| V7 | 两个 Adapter 结果结构完全一致（字段不缺、类型正确） | models 校验 |

### 8.2 Phase 2 验收

- 跨源去重有效（同论文在 arXiv+Scholar 只出现一次）
- 排序权重可在配置文件调整，调整后顺序变化符合预期
- GitHub 支持按文件名/内容搜索代码文件并下载
- 支持多轮对话 refine

### 8.3 最终验收（Phase 3+）

- 新增一个 Adapter（如 HuggingFace）≤ 1 人日完成且不改动框架代码
- 全流程 P95 耗时 ≤ 15s（不含大文件下载）
- 所有失败路径都有日志可查

---

## 9. 风险与应对

| 风险 | 概率 | 影响 | 应对 |
|---|---|---|---|
| GitHub Search API 未认证限流过严 | 高 | 中 | 每人申请免费 Token；测试全部 mock |
| Google Scholar 反爬严格 | 高 | 中 | Scholar 降为 P1 且用官方库/SerpAPI；arXiv 保底 |
| 新人进度不及预期 | 中 | 高 | 任务预留缓冲；负责人 W1 把骨架做厚，降低上手门槛；随时砍 P2 任务 |
| PDF 解析质量参差（扫描版论文） | 中 | 低 | 解析失败降级为「仅下载不解析」，不阻塞主流程 |
| 需求蔓延（想加 Web UI、用户系统） | 中 | 中 | 严守 §1.3 非目标清单，新需求进 Issue 池下版本评审 |
| 大文件下载失败率高 | 低 | 中 | 断点续传 + 失败重试 + 失败日志 |

---

## 10. 附录

### 10.1 新人学习资料清单（负责人需在 W1 发出）

- Python asyncio 入门：官方文档 asyncio 章节
- httpx 异步请求：https://www.python-httpx.org/async/
- arXiv API 文档：https://info.arxiv.org/help/api/
- GitHub REST API：https://docs.github.com/rest
- typer 教程：https://typer.tiangolo.com/
- pytest + respx：各自官方 quickstart
- 内部：ExampleAdapter 源码 + 本 PRD

### 10.2 配置文件示例（settings.yaml）

```yaml
download:
  base_dir: "./downloads"
  max_concurrent: 3
  max_retries: 3
  timeout_seconds: 60

ranking:
  weights:
    source: 0.4        # 来源可信度
    recency: 0.3       # 时效性（按年份衰减）
    popularity: 0.3    # 引用数/stars 对数归一

sources:
  arxiv:
    enabled: true
    rate_limit_qps: 1
  github:
    enabled: true
    rate_limit_qps: 2
```

### 10.3 需求-任务追踪矩阵

| PRD 需求 | Phase | 任务编号 | Owner |
|---|---|---|---|
| F3.1 arXiv | P1 | A-1/A-2 | 成员A |
| F3.2/3.3/3.4 GitHub | P1 | B-1/B-2/B-3 | 成员B |
| F5.x 下载管理 | P1 | B-5 | 成员B |
| F1.x 意图解析 | P1 | X-1 | 负责人 |
| F2.x 路由 | P1 | 公共 | 负责人 |
| F4.x 结果处理 | P2 | X-2 | 负责人+A/B |
| F3.5 Scholar | P2 | A-4 | 成员A |
| F3.6 Web | P2 | A-5 | 成员A |
| F5.5 整仓/ZIP | P2 | B-4 | 成员B |
| F6.x 内容解析 | P3 | A-6/B-6 | A/B |
| F7.x 交付交互 | P1/P2 | X-3/X-4 | 负责人/交叉 |

---

*本文档为团队唯一事实来源（Single Source of Truth）。任何范围变更需经全员在 Issue 中讨论并更新本文档后生效。*
