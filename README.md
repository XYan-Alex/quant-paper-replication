# 量化论文复现 · quant-paper-replication

把 <!-- N -->8<!-- /N --> 篇量化投资论文（持续更新）从数据到代码复现了一遍，全部代码、每次运行的结果都在这里。配套视频见抖音 / B 站账号 **量化舵**。

> 每一篇都把论文报告的数字和我们复现的数字并排列出，口径上的差异逐条说明，详见各文件夹。所有数字都是历史模拟，非实盘，不构成任何投资建议。

## 一览

<!-- PAPERS:START -->
| # | 论文 | 来源 | 方向 |
|---|---|---|---|
| [01](01_dmfm) | Deep Multi-Factor Model | arXiv:2210.12462 | 深度学习多因子选股 |
| [02](02_alpha_gpt) | Alpha-GPT | arXiv:2308.00016 | 大模型 + 遗传搜索挖因子 |
| [03](03_quanta_alpha) | QuantaAlpha | arXiv:2602.07085 | 因子进化框架（精简版） |
| [04](04_etf_rotation) | ETF 行业轮动 | 汉斯《金融》2026 | 动量 × 风险优化 |
| [05](05_smart_tangency) | Smart Tangency Portfolio | IJFS 2025 | 强化学习选调仓时点 |
| [06](06_last_hour_momentum) | Last hour momentum | CFRI 2021 | 尾盘半小时动量 |
| [07](07_neural_shrinkage) | Neural Shrinkage | arXiv:2608.30446 | 神经网络估计协方差 |
| [08](08_sharpe_driven) | Sharpe-Driven Stock Selection | arXiv:2511.13251 | 滚动夏普选股 |
| [附](overfitting_check) | PBO、DSR 过拟合检验 | — | 回测稳健性检验 |
<!-- PAPERS:END -->

每个文件夹里有单独的 README：论文在做什么、每个脚本的作用、怎么运行、结果表、与论文口径的对照。

## 目录
<!-- TREE:START -->
```
01_dmfm/                 Deep Multi-Factor Model
02_alpha_gpt/            Alpha-GPT
03_quanta_alpha/         QuantaAlpha
04_etf_rotation/         ETF 行业轮动
05_smart_tangency/       Smart Tangency Portfolio
06_last_hour_momentum/   Last hour momentum
07_neural_shrinkage/     Neural Shrinkage
08_sharpe_driven/        Sharpe-Driven Stock Selection
overfitting_check/        PBO / DSR 过拟合检验
results/                 每篇的结果文件（JSON / CSV）
data/                    数据目录（不进版本库，见 data/README.md）
papers.json              论文清单（README 一览表和指引 PDF 都由它生成）
tools/build.py           重新生成一览表和 docs/guide.pdf
_template/               新增论文时用的 README 模板
```
<!-- TREE:END -->

## 怎么跑
```bash
git clone https://github.com/XYan-Alex/quant-paper-replication.git
cd quant-paper-replication
pip install -r requirements.txt
```
然后按 [`data/README.md`](data/README.md) 准备数据，再进各文件夹照 README 里的命令运行。数据默认放在 `data/`，也可以 `export QPR_DATA=/别的目录`。

- 环境：Python 3.10 以上，全部在 2 核 CPU 上跑通，不需要显卡。
- 01 的模型训练最慢（几小时）；05 约 40 分钟；07 约 1 小时；其余几分钟。

## 通用口径
- A 股日线来自公开的众包数据集，股票池是沪深300 / 中证500 / 中证1000 的历史成分股（含已退市股票）。
- 回测都在信号之后的下一个交易日成交，计入交易成本（各篇成本写在各自 README 里），基准是价格指数。
- “论文之后”的区间一律单独列出：那段行情论文没见过，我们调参时也没用过。

## 新增一篇论文（仓库持续更新）
1. 新建文件夹 `NN_简称/`（编号顺延），脚本放进去，数据路径统一用 `QPR_DATA`；
2. 照 [`_template/README.md`](_template/README.md) 写这一篇的 README；结果文件放 `results/NN_简称/`；
3. 在 [`papers.json`](papers.json) 末尾加一条；
4. 运行 `python3 tools/build.py`，自动更新上面的一览表、目录和 [`docs/guide.pdf`](docs/guide.pdf)（入门指引）。

## 说明
- 代码以“能复查”为目标写的，单文件、依赖少，不是工程化的框架。
- 02、03 里“大模型”这一步（把想法翻成公式、看结果重写公式）由 Claude 完成，产出的公式都保存在仓库里。
- 发现问题欢迎提 issue。
