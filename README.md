# RAG 与 CRAG 对比项目

本项目实现**标准 RAG（检索增强生成）**与 **CRAG（Corrective RAG，纠正式 RAG）** 的完整流程，支持从 PDF 解析、向量索引构建到问答的端到端流水线，并可通过命令行对比两种方案的检索与生成过程及最终答案。

---

## 目录

- [项目概述](#项目概述)
- [系统架构与流程](#系统架构与流程)
- [环境要求](#环境要求)
- [安装与配置](#安装与配置)
- [环境变量说明](#环境变量说明)
- [使用说明](#使用说明)
- [项目结构](#项目结构)
- [核心模块说明](#核心模块说明)
- [RAG 与 CRAG 的区别](#rag-与-crag-的区别)
- [常见问题](#常见问题)

---

## 项目概述

- **RAG**：根据用户问题从本地向量库检索文档，将检索结果作为上下文交给大模型生成答案。
- **CRAG**：在 RAG 基础上增加「检索质量评估」与「纠正」逻辑：若评估发现检索文档与问题相关性不足，会先**重写问题**并调用**联网搜索（Tavily）**，再将本地检索结果与网络结果一起用于生成，从而在文档不足或不够相关时自动补充外部知识。

本仓库同时提供：

1. **PDF → Markdown**：基于 MinerU + VLM 的 PDF 解析（保留标题、正文、图片、表格、公式）。
2. **Markdown → 向量库**：按标题层级与长度切分 Markdown，用 BGE 做向量化并写入 ChromaDB。
3. **问答入口**：通过 `cli.py` 一次提问可同时跑 RAG 与 CRAG，便于对比中间步骤与最终回答。

默认场景面向**铁路相关知识**文档，可通过修改 Prompt 与数据源适配其他领域。

---

## 系统架构与流程

### 整体流水线

```
┌─────────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│  PDF 文档       │ ──► │ 0_pdf2md_by_mineru  │ ──► │ Markdown 文件    │
│  (原始资料)     │     │ (MinerU + VLM 解析) │     │ (含图/表/公式)   │
└─────────────────┘     └─────────────────────┘     └────────┬─────────┘
                                                              │
                                                              ▼
┌─────────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│  用户问题       │ ──► │ 1_create_index      │ ◄── │ 按标题切分 + BGE  │
│  (CLI/交互)     │     │ (BGE + ChromaDB)    │     │ 存入向量库        │
└────────┬────────┘     └─────────────────────┘     └──────────────────┘
         │
         ├───────────────────────────────────────────────────────────────┐
         │                                                               │
         ▼                                                               ▼
┌─────────────────────┐                                    ┌─────────────────────┐
│  RAG 流程           │                                    │  CRAG 流程          │
│  检索 → 生成        │                                    │  检索 → 评估 → 决策  │
│  (rag_pipeline)     │                                    │  → [重写+联网] → 生成│
└─────────────────────┘                                    └─────────────────────┘
```

### RAG 流程（简化）

1. **检索**：用 BGE 对问题做向量化，在 ChromaDB 中取 Top-K 文档。
2. **生成**：将检索到的文档拼成上下文，与问题一起交给 DeepSeek 生成答案。

### CRAG 流程（LangGraph）

1. **检索**：与 RAG 相同，从 ChromaDB 取 Top-K 文档。
2. **评估**：用本地 LLM 对每条文档做「是否与问题相关」的二元评分（yes/no）。
3. **决策**：
   - 若**全部相关**：直接进入「生成」。
   - 若**存在不相关**：先「重写问题」→「Tavily 联网搜索」→ 将网络结果与当前文档合并后再「生成」。
4. **生成**：与 RAG 相同，基于最终上下文 + 问题由 DeepSeek 生成答案。

---

## 环境要求

- **Python**：建议 3.10+
- **GPU**：PDF 解析（MinerU/VLM）与 BGE 向量化建议使用 CUDA GPU；仅运行 RAG/CRAG 推理时可用 CPU（需在代码中调整 `device`）。
- **依赖**：见 `requirements.txt`，主要包含：
  - LangChain / LangGraph
  - ChromaDB、BGE（HuggingFace Embeddings）
  - MinerU、vLLM、PyMuPDF（PDF 解析）
  - OpenAI 兼容 API（DeepSeek、本地评估模型）
  - Tavily（联网搜索）
  - python-dotenv

---

## 安装与配置

### 1. 克隆与依赖

```bash
cd RAG_and_CRAG
pip install -r requirements.txt
```

（若使用 conda/venv，请先激活对应环境。）

### 2. 环境变量

在项目根目录创建 `.env` 文件，填入所需变量。完整说明见 [环境变量说明](#环境变量说明)。

### 3. 模型与数据准备

- **PDF 解析（可选）**：配置 `MINERU_MODEL_PATH`（MinerU 模型路径，如 `../.cache/modelscope/hub/models/OpenDataLab/MinerU2.5-2509-1.2B`），并设置 `PDF_GLOB_PATH`、`MD_GLOB_PATH` 指定 PDF 输入与 Markdown 输出目录。
- **向量索引**：配置 `BGE_MODEL_PATH`（如 `BAAI/bge-small-zh-v1.5` 或本地路径）、`VECTOR_DB_DIR`（Chroma 持久化目录）、`MD_GLOB_PATH`（Markdown 根目录，用于建库）。
- **评估模型**：配置 `LOCAL_LLM_BASE_URL`、`LOCAL_LLM_MODEL_NAME`、`LOCAL_LLM_API_KEY`（OpenAI 兼容接口，用于文档相关性打分）。
- **生成/重写**：配置 `DEEPSEEK_NAME`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_API_KEY`。
- **联网搜索（CRAG）**：配置 `TAVILY_API_KEY`。

---

## 环境变量说明

| 变量名 | 用途 | 示例/说明 |
|--------|------|-----------|
| `MINERU_MODEL_PATH` | MinerU 模型路径（PDF 解析） | `../.cache/modelscope/hub/models/OpenDataLab/MinerU2.5-2509-1.2B` |
| `PDF_GLOB_PATH` | PDF 文件匹配路径 | `./pdfs/*.pdf` |
| `MD_GLOB_PATH` | Markdown 输出/建库根目录 | `./mineru_OCRpdf` |
| `BGE_MODEL_PATH` | BGE 向量模型名或路径 | `BAAI/bge-small-zh-v1.5` |
| `VECTOR_DB_DIR` | ChromaDB 持久化目录 | `./vector_db/railway_db` |
| `LOCAL_LLM_BASE_URL` | 评估用 LLM API 地址 | `http://localhost:8000/v1` |
| `LOCAL_LLM_MODEL_NAME` | 评估用模型名 | `your-local-model` |
| `LOCAL_LLM_API_KEY` | 评估用 API Key（可为空） | `sk-xxx` 或留空 |
| `DEEPSEEK_NAME` | 生成/重写用模型名 | `deepseek-chat` |
| `DEEPSEEK_BASE_URL` | DeepSeek API 地址 | `https://api.deepseek.com` |
| `DEEPSEEK_API_KEY` | DeepSeek API Key | `sk-xxx` |
| `TAVILY_API_KEY` | Tavily 搜索 API Key（CRAG 联网） | 在 Tavily 官网申请 |
| `RETRIEVE_TOP_K` | 检索返回文档数 | 默认 `3` |
| `WEB_SEARCH_MAX_RESULTS` | 联网搜索返回条数 | 默认 `3` |

---

## 使用说明

### 步骤 0：PDF 转 Markdown

若已有 PDF 需要解析为 Markdown：

```bash
python 0_pdf2md_by_mineru.py
```

- 会读取 `PDF_GLOB_PATH` 下的 PDF，输出到 `MD_GLOB_PATH` 对应目录。
- 自动跳过仅 1 页或 0 页的文件，并会识别「目次」页并跳过。
- 需要 GPU 与 MinerU 模型，耗时较长，适合批量预处理。

### 步骤 1：构建向量索引

在已有 Markdown 文件（或完成步骤 0 后）执行：

```bash
python 1_create_index.py
```

- 会扫描 `MD_GLOB_PATH` 下所有 `.md` 文件。
- 按 `#`–`####` 标题层级切分，再按 600 字符、100 重叠做二次切分，并写入 BGE 向量与 ChromaDB（`VECTOR_DB_DIR`）。
- 脚本末尾包含一次检索测试，可确认建库是否正常。

### 步骤 2：运行 RAG 与 CRAG 对比

**推荐方式：命令行一次提问同时跑 RAG + CRAG**

```bash
# 单次提问
python cli.py -q "产品出厂前，包装上的合格证应包含哪些内容？"

# 交互模式：多次输入问题，每次都会跑 RAG 和 CRAG
python cli.py -i
```

- 会先输出 RAG 的检索条数、生成结果，再输出 CRAG 的检索、评估、决策、是否重写与联网、以及最终回答。
- 便于对比「仅用本地文档」与「在文档不佳时自动联网」的差异。

**仅运行 CRAG（单流程）**

```bash
python crag.py
```

- 使用脚本内默认问题；若需改问题，可编辑 `crag.py` 末尾的 `question` 或改为从参数/标准输入读取。

**通过标准输入传题（便于管道）**

```bash
echo "你的问题" | python cli.py
```

---

## 项目结构

```
RAG_and_CRAG/
├── .env                    # 环境变量（需自行创建，勿提交密钥）
├── README.md               # 本说明
├── requirements.txt        # Python 依赖
│
├── 0_pdf2md_by_mineru.py   # PDF → Markdown（MinerU + VLM）
├── 1_create_index.py       # Markdown 切分 + BGE + ChromaDB 建库
├── cli.py                  # 命令行入口：RAG + CRAG 对比
├── crag.py                 # 仅运行 CRAG（默认问题）
├── crag_by_myself.py       # 早期 CRAG 图实现（可作参考，根据这个文件让cursor进行代码优化）
│
├── config.py               # 统一从 .env 读取配置
├── models.py               # 图状态等数据结构（如 GraphState）
├── retriever.py            # BGE + Chroma 检索器
├── chains.py               # 评估链、生成链、问题重写链
├── rag_pipeline.py         # 标准 RAG：检索 → 生成
├── crag_graph.py           # CRAG LangGraph：检索 → 评估 → 决策 → [重写+联网] → 生成
│
├── mineru_OCRpdf/          # MinerU 输出目录（示例，可由 MD_GLOB_PATH 指定）
└── vector_db/              # ChromaDB 持久化目录（可由 VECTOR_DB_DIR 指定）
    └── railway_db/
```

---

## 核心模块说明

| 文件 | 作用 |
|------|------|
| `config.py` | 从 `.env` 加载 `BGE_MODEL_PATH`、`VECTOR_DB_DIR`、本地 LLM、DeepSeek、Tavily、`RETRIEVE_TOP_K`、`WEB_SEARCH_MAX_RESULTS` 等，供各模块使用。 |
| `retriever.py` | 使用 `config` 中的 BGE 与 Chroma 路径，构造 `get_embeddings()`、`get_vector_store()`、`get_retriever(search_kwargs={"k": RETRIEVE_TOP_K})`。 |
| `chains.py` | **评估链**：本地 LLM，输入文档+问题，输出 yes/no；**生成链**：DeepSeek，输入 context+问题，输出答案；**重写链**：DeepSeek，输入问题，输出改写后问题。 |
| `models.py` | 定义 `GraphState`（question, generation, web_search, documents），供 LangGraph 使用。 |
| `rag_pipeline.py` | `run_rag(question)`：调用 retriever 检索 → 拼 context → 调用 RAG 生成链，返回答案字符串。 |
| `crag_graph.py` | 定义 retrieve、grade_documents、transform_query、web_search、generate 节点及条件边，`run_crag(question)` 编译图并流式执行，返回最终生成结果。 |

---

## RAG 与 CRAG 的区别

| 维度 | RAG | CRAG |
|------|-----|------|
| 检索后处理 | 直接使用 Top-K 文档 | 先对每条文档做相关性评估（yes/no） |
| 文档质量差时 | 仍只用本地检索结果，易答非所问或空洞 | 触发「问题重写 + Tavily 联网」，用网络结果补充后再生成 |
| 实现 | 线性：检索 → 生成 | 图：检索 → 评估 → 条件分支 → 可选重写+联网 → 生成 |
| 适用场景 | 文档质量高、领域内问题 | 文档不全或噪声多、需要外部知识兜底 |

---

## 常见问题

**Q: 未配置 Tavily，CRAG 会报错吗？**  
A: 会。CRAG 在决策为「需要联网」时会调用 Tavily；若未配置 `TAVILY_API_KEY` 或 Key 无效，将报错。可暂时在 `crag_graph.py` 中注释或跳过 `web_search` 相关逻辑做本地测试。

**Q: 评估模型必须用本地 LLM 吗？**  
A: 不必。只要提供 OpenAI 兼容的 API（`LOCAL_LLM_BASE_URL` + `LOCAL_LLM_MODEL_NAME` + `LOCAL_LLM_API_KEY`），可以是本地部署或其它兼容服务。

**Q: 如何只做 RAG、不做 CRAG？**  
A: 直接调用 `rag_pipeline.run_rag(question)`，或写一个只跑 RAG 的小脚本；CLI 设计为「一次同时跑两者」便于对比。

**Q: 如何更换领域（非铁路）？**  
A: 修改 `chains.py` 中生成链与重写链的 system/user prompt，并用自己的 PDF/Markdown 重新跑步骤 0 和 1 即可。

**Q: 向量库已存在，再跑 `1_create_index.py` 会怎样？**  
A: Chroma 的 `from_documents` 会向已有库追加文档；若需完全重建，请先删除 `VECTOR_DB_DIR` 目录再运行。

---

## 许可证与致谢

- 本项目使用 LangChain / LangGraph、Chroma、MinerU、BGE、DeepSeek API、Tavily 等，请遵循各自许可协议。
- CRAG 思路参考相关论文与 LangChain/LangGraph 官方示例。
- 参考代码：https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_crag.ipynb
