# RoTE 在 ReChorus 中的 CPU 复现（理论课大作业，25 组）

组员：史熠鑫（24350124）、王宸（23330123）。

本仓库基于 [ReChorus 2.0](https://github.com/THUwangcy/ReChorus/tree/c164ec4303cc20ddcfbd1b57de366a481811d1e5) 提交 `c164ec4303cc20ddcfbd1b57de366a481811d1e5`，复现 [RoTE 论文](https://arxiv.org/abs/2604.13389) 中的日历时间旋转注意力序列推荐模型，并与 SASRec、GRU4Rec 在两套数据上比较。上游说明见 [README-ReChorus-upstream.md](README-ReChorus-upstream.md)，许可证见 [LICENSE](LICENSE)。RoTE 模型依据论文独立编写，[作者代码](https://github.com/XiaoLongtaoo/RoTE) 仅用于核对方法。

## 文件

- `src/models/sequential/RoTE.py`：通过 `--model_name RoTE` 接入 ReChorus 的 `SeqReader`、`BaseRunner` 及候选排序损失。
- `scripts/prepare_movielens_topk.py`：将 MovieLens-1M 评分转换为 Top-K 留一数据。
- `scripts/prepare_grocery_unique.py`：保留框架 Grocery 正样本，固定去重负样本。
- `scripts/audit_data.py`：检查分割、时间、ID 和候选列表。
- `scripts/run_cpu_experiments.py`：单轮检查和三种子正式实验，保存每次命令、日志、参数量及耗时。
- `scripts/summarize_results.py`：验证 18 次正式运行并汇总均值与标准差。
- `tests/`：日期、旋转、时间敏感性、反向传播、数据准备和 ReChorus 集成测试。
- `results/`：正式结果、原始日志和数据审计；`report/`：中文报告。

实际运行时 MovieLens 模型进程曾并行执行，CPU 线程数与资源竞争细节见 `results/execution_context.md`；所记耗时不用于受控效率排名。

## 环境与复现

使用 Python 3.10。本项目的实验环境为 Windows、Python 3.10.11、PyTorch 2.2.2，全部使用 CPU。不要直接安装上游 `requirements.txt`，其中有 `cudatoolkit`、`pickle` 等非 pip 依赖。在仓库根目录运行 PowerShell：

```powershell
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
$env:PYTHONPATH = 'src;.'
.venv\Scripts\python.exe -m pytest tests -q
```

框架自带 `data/Grocery_and_Gourmet_Food/`。从 [GroupLens MovieLens-1M](https://grouplens.org/datasets/movielens/1m/) 下载并解压 `ratings.dat`，放到 `raw-data/ml-1m/ratings.dat`（也可使用仓库外路径），然后执行：

```powershell
.venv\Scripts\python.exe scripts/prepare_movielens_topk.py --ratings raw-data/ml-1m/ratings.dat --output data/ML_1MTOPK_RoTE
.venv\Scripts\python.exe scripts/prepare_grocery_unique.py --source data/Grocery_and_Gourmet_Food --output data/Grocery_RoTE_99Unique
.venv\Scripts\python.exe scripts/audit_data.py data/ML_1MTOPK_RoTE --output results/audit_ml1m.json
.venv\Scripts\python.exe scripts/audit_data.py data/Grocery_RoTE_99Unique --output results/audit_grocery.json
.venv\Scripts\python.exe scripts/run_cpu_experiments.py --mode smoke
.venv\Scripts\python.exe scripts/run_cpu_experiments.py --mode full
.venv\Scripts\python.exe scripts/summarize_results.py
.venv\Scripts\python.exe scripts/build_report_pdf.py
```

`raw-data` 和派生数据都不提交。数据脚本默认固定候选抽样种子 2026；三个模型及训练种子使用每套数据的相同候选集。运行器会跳过已有完整 JSON 记录；重跑某次实验时先移走对应 `results/runs/full/` 文件。

## 数据与实验协议

MovieLens-1M 保留评分 ≥4 的正反馈，反复执行 5-core 过滤，按每个用户的时间顺序留最后两次交互为验证、测试。过滤留出物品未在训练中出现的用户；ID 从 1 连续映射。验证/测试各固定抽取 99 个不重复、未被该用户交互的负样本。**这是本项目构造的留一协议，不是直接采用框架笔记本的 MovieLens 划分。**

框架 Grocery 的训练及验证/测试正样本保持不变。源候选列表存在重复负样本，因此重新按固定种子抽取每行 99 个不同的未交互物品。原 Grocery 有少量留出正物品不在训练中，数量见数据审计文件；这是结果的一项局限。

三模型使用 CPU、历史长度 20、嵌入维度 16、批量 256、学习率 0.001、最多 8 轮、验证 NDCG@10 选最佳轮次、早停耐心 3、随机种子 0/1/2。RoTE 和 SASRec 都是 1 层 2 头，GRU4Rec 隐层 16。测试指标为 100 候选下 HR@5/10、NDCG@5/10，报告三次均值与样本标准差。每次的完整命令、参数量和训练总耗时保存在 `results/runs/full/*.json`。

RoTE 将 Unix 秒时间按 UTC 拆为自 1970 年起累计的年、月、日序号，对注意力 Query/Key 每个头的相邻维度分别旋转，以基数 10^6、10^4、10^2 计算频率，以 1.5、1.0、0.5 加权融合。Value、候选打分和训练接口遵循 ReChorus。论文与本项目使用不同数据及评价协议，**本课程实验数值不能与论文表格数值直接对齐**。

## 来源

- Zhang 等：*RoTE: Coarse-to-Fine Multi-Level Rotary Time Embedding for Sequential Recommendation*，SIGIR 2026，[论文](https://arxiv.org/abs/2604.13389)，[作者代码](https://github.com/XiaoLongtaoo/RoTE)。
- [ReChorus 官方仓库](https://github.com/THUwangcy/ReChorus)。
- [GroupLens MovieLens-1M](https://grouplens.org/datasets/movielens/1m/)。
