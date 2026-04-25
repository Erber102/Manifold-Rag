# Manifold-RAG 本地修改总结

对比原始仓库 `https://github.com/Erber102/Manifold-Rag` (main branch)，本地做了以下修改。

---

## 一、修改的原有文件

### `experiments/run_retrieval.py`（342行 → 651行，核心重写）

| 项目 | 原始版本 | 修改后版本 |
|------|----------|------------|
| 数据集 | `["scifact", "fiqa"]` | `["wiki"]`（自建层级语料库）|
| 数据路径 | `"datasets"` | `"data"` |
| 多 seed | 无 | `SEEDS = [42, 123]`，自动计算 mean±std |
| 负样本 | 纯随机 | 基于 section title 的层级 hard negative（同标题跨分支）|
| 正则化 | 无 | 层级 norm 正则化（category→0.20，article→0.45，section→0.70，λ=0.2）|
| 梯度裁剪 | 无 | `max_norm=1.0` |
| Warm-start | 无 | Poincaré / Product head 从 Euclidean 权重初始化 |
| 训练 epoch | 所有方法均 20 | Euclidean=20，Product=30，Poincaré=50 |
| PoincareHead.scale | 0.1 | 0.3 |
| ProductHead.scale_h | 0.1 | 0.3 |
| 评测 | 整体指标 | 整体 + 按 QA 类型细分（generalize / hierarchy_navigation / hierarchy_disambiguation）|
| 新增函数 | — | `set_seed()`, `aggregate_seed_runs()`, `build_hard_neg_index()`, `load_query_types()`, `build_level_targets()`, `eval_with_types()` |

### `baseline/bm25_baseline.py`（小幅修改）
- 数据集：`["scifact", "fiqa"]` → `["wiki"]`
- 保存逻辑：改为追加合并（不覆盖已有结果）

### `baseline/sbert_baseline.py`（小幅修改）
- 数据集：`["scifact", "fiqa"]` → `["wiki"]`
- 保存逻辑：改为追加合并（不覆盖已有结果）

### `requirements.txt`（更新依赖版本）
- 原版：无版本号，10 个包，`rank_bm25` 命名有误
- 新版：全部加 `>=` 版本约束，新增 `faiss-cpu`, `pandas`, `seaborn`, `requests`，修正为 `rank-bm25`

---

## 二、新增文件

### `wiki_mine.py` — Wikipedia 层级语料库爬取脚本
- 从 8 个 seed 类别出发（Biology, Physics, Mathematics, Computer Science, History, Philosophy, Chemistry, Economics）
- 通过 Wikipedia Category API 递归爬取子类别（MAX_DEPTH=4，MAX_SUBCATS_PER_CAT=5）
- 每个 seed 最多 100 篇文章（MAX_ARTICLES_PER_SEED=100），总量上限 1000 篇
- 为每个 chunk 记录完整层级路径（category → subcategory → article → section）
- 输出 BEIR 格式 `corpus.jsonl`，含 `metadata.hierarchy`、`metadata.type`、`metadata.level`

### `generate_hard_qa.py` — 层级感知 QA pair 生成脚本
- 生成 4 种 QA 类型：
  - `generalize`（section → category 文档，测试泛化；测试集占比 70%）
  - `hierarchy_navigation`（article → category 文档，测试层级导航；测试集 50%）
  - `hierarchy_disambiguation`（仅用层级路径作 query，不含文章名，测试消歧；测试集 10%）
  - `sibling`（仅用于训练，测试集比例 0）
- `_clean_snippet()` 去除 query 中的文章/类别名（防止信息泄露）
- 输出 `queries.jsonl`、`qrels/train.tsv`、`qrels/test.tsv`（BEIR 格式）

### `experiments/run_baselines.py` — 独立 baseline 脚本
- Raw SBERT（all-MiniLM-L6-v2，无微调）+ 余弦相似度
- BM25（rank-bm25）词法检索
- 同样计算 per-type 指标，结果 append 至 `results/manifold_results.json`
- 与 `run_retrieval.py` 完全分离，可单独运行（~2 分钟）

### `data/wiki/` — 自建 BEIR 格式数据集
```
data/wiki/
├── corpus.jsonl         # ~1000 篇文章的层级语料库
├── queries.jsonl        # 965 条 QA pair（含 qa_type metadata）
└── qrels/
    ├── train.tsv        # 训练集相关性标注
    └── test.tsv         # 测试集相关性标注
```

### `results/manifold_results.json` — 更新的实验结果
- 包含 2 个 seed（42, 123）逐条结果 + 跨 seed 聚合（mean±std）
- 包含 per-type breakdown

---

## 三、未修改的文件

- `test_code/`（4 个测试脚本）
- `iclr_template/`（ICLR 2026 模板）
- `data/download_data_script.py`
- `README.md`
- `report/`（LaTeX 报告，由其他组员维护）

---

## 四、实验结论（两 seed 验证，std < 0.007）

| Method | Overall NDCG@10 | disambiguation NDCG@10 | navigation NDCG@10 | generalize NDCG@10 |
|--------|:---------:|:----------------------:|:------------------:|:------------------:|
| BM25 | — | — | — | — |
| SBERT raw | — | — | — | — |
| Euclidean-64 | 0.5777 ± 0.003 | 0.452 ± 0.001 | 0.580 ± 0.002 | 0.611 ± 0.006 |
| Poincaré-64 | 0.4921 ± 0.001 | 0.284 ± 0.003 | 0.488 ± 0.002 | 0.551 ± 0.002 |
| **Product-64** | 0.5417 ± 0.001 | **0.543 ± 0.007** | 0.470 ± 0.003 | 0.565 ± 0.001 |

**核心发现**：Product 流形在 `hierarchy_disambiguation` 上显著优于 Euclidean（+9.1 NDCG@10），信噪比 13:1，结论稳健。BM25 / SBERT raw 的数值在运行 `python experiments/run_baselines.py` 后填入。
