# data/

这个目录不进版本库。所有脚本默认把数据读写到这里；也可以用环境变量 `QPR_DATA=/你的/数据目录` 指到别处。

| 数据 | 来源 | 怎么拿 |
|---|---|---|
| A 股日线（含三大指数历史成分） | [chenditc/investment_data](https://github.com/chenditc/investment_data/releases) 发布的 `qlib_bin.tar.gz` | 下载后解压到 `data/qlib_bin/`（里面应有 `calendars/ features/ instruments/`），再运行 `python3 01_dmfm/prices.py` 生成 `prices.npz` |
| 季度业绩报表、资产负债表 | 东方财富（经 akshare） | `python3 01_dmfm/fetch_fund.py` |
| 行业 ETF 日线 | 新浪（经 akshare） | `python3 04_etf_rotation/fetch_etf.py` |
| 美股行业 ETF、道指成分股日线 | Yahoo Finance | `python3 05_smart_tangency/fetch_us.py` |
| 30 分钟 K 线 | 新浪（经 akshare，只有最近约一年） | `python3 06_last_hour_momentum/fetch_30min.py` |

行情和财报数据的版权归各数据提供方，本仓库不再分发。
