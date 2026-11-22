"""Last hour momentum in the Chinese stock market（Yang et al., CFRI 2021）复现。
论文：第 6（7）个半小时的涨跌，能预测第 7（8）个半小时；据此做尾盘择时（只做多）。论文样本是 2012—2017 年。
我们只能拿到公开接口最近约一年的 30 分钟 K 线（2025-11 至 2026-11，约 245 个交易日），所以这是“论文之后的样本外检验”，不是原样本复现。
先运行 fetch_30min.py。"""
import os, json, numpy as np, pandas as pd
DATA = os.environ.get("QPR_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "06_last_hour_momentum"); os.makedirs(OUT, exist_ok=True)
SYMS = {"sh000300": "沪深300指数", "sh510300": "沪深300ETF", "sh000905": "中证500指数", "sh510500": "中证500ETF", "sh000001": "上证指数", "sz399001": "深证成指"}
def halfhours(sym):
    d = pd.read_parquet(os.path.join(DATA, "m30", sym + ".parquet")).sort_values("day"); d["date"] = d.day.dt.normalize(); d["k"] = d.groupby("date").cumcount() + 1
    n = d.groupby("date").k.max(); d = d[d.date.isin(n[n == 8].index)]                         # 只留 8 根齐全的交易日
    c = d.pivot(index="date", columns="k", values="close"); prev = c[8].shift(1)
    r = pd.DataFrame({1: c[1] / prev - 1, **{k: c[k] / c[k - 1] - 1 for k in range(2, 9)}}).dropna(); r.columns = [f"r{k}" for k in r.columns]
    return r
def ols_nw(y, X, lags=5):
    """OLS + Newey-West 标准误，返回系数、t 值、R²"""
    X = np.column_stack([np.ones(len(y)), X]); b = np.linalg.lstsq(X, y, rcond=None)[0]; e = y - X @ b; n, k = X.shape
    S = (X * e[:, None]).T @ (X * e[:, None])
    for l in range(1, lags + 1):
        G = (X[l:] * e[l:, None]).T @ (X[:-l] * e[:-l, None]); S += (1 - l / (lags + 1)) * (G + G.T)
    XtX = np.linalg.inv(X.T @ X); V = XtX @ S @ XtX; return b, b / np.sqrt(np.diag(V)), 1 - e.var() / y.var()
def stats(x):
    x = np.asarray(x); return dict(ann=round(x.mean() * 244 * 100, 2), vol=round(x.std() * np.sqrt(244) * 100, 2), sharpe=round(x.mean() / x.std() * np.sqrt(244), 2) if x.std() > 0 else None, t=round(x.mean() / x.std() * np.sqrt(len(x)), 2) if x.std() > 0 else None)
if __name__ == "__main__":
    res = {}
    for sym, name in SYMS.items():
        r = halfhours(sym); y8, y7 = r.r8.values, r.r7.values; o = dict(days=len(r), start=str(r.index[0])[:10], end=str(r.index[-1])[:10])
        for tag, y, X in (("r8~r7", y8, r[["r7"]].values), ("r8~r6+r7", y8, r[["r6", "r7"]].values), ("r7~r6", y7, r[["r6"]].values), ("r8~r1（首半小时，Gao 等 2018）", y8, r[["r1"]].values)):
            b, t, r2 = ols_nw(y, X); o[tag] = dict(beta=[round(float(v), 4) for v in b[1:]], t=[round(float(v), 2) for v in t[1:]], r2=round(float(r2) * 100, 2))
        # 样本外 R²（前一半估计，递推预测后一半；基准 = 历史均值）
        h = len(r) // 2; pe, pm = [], []
        for i in range(h, len(r)):
            b = np.linalg.lstsq(np.column_stack([np.ones(i), r.r7.values[:i]]), y8[:i], rcond=None)[0]; pe.append(y8[i] - b[0] - b[1] * r.r7.values[i]); pm.append(y8[i] - y8[:i].mean())
        o["r2_os_r8~r7"] = round((1 - np.sum(np.square(pe)) / np.sum(np.square(pm))) * 100, 2)
        e7 = np.where(r.r6 > 0, r.r7, 0.0); e8 = np.where(r.r7 > 0, r.r8, 0.0)                    # 择时：信号为正才做多，下一根半小时收盘平仓
        o["策略"] = {"η(r7)：r6>0 则持有第7个半小时": dict(**stats(e7), win=round(float((r.r7[r.r6 > 0] > 0).mean()) * 100, 1), trades=int((r.r6 > 0).sum())),
                    "η(r8)：r7>0 则持有第8个半小时": dict(**stats(e8), win=round(float((r.r8[r.r7 > 0] > 0).mean()) * 100, 1), trades=int((r.r7 > 0).sum())),
                    "η(r7,8)：两者叠加": stats(e7 + e8),
                    "不择时：每天都持有最后一小时": stats(r.r7 + r.r8),
                    "买入持有（全天）": stats((1 + r).prod(axis=1) - 1)}
        for c_ in (0.00005, 0.0001):                                                             # 单边成本 0.5bp / 1bp
            o["策略"][f"η(r7,8) 扣单边 {c_ * 1e4:.1f}bp"] = stats(e7 + e8 - 2 * c_ * ((r.r6 > 0).astype(float) + (r.r7 > 0).astype(float)).values + 2 * c_ * ((r.r6 > 0) & (r.r7 > 0)).values)
        res[name] = o; print(name, json.dumps(o, ensure_ascii=False)[:900], flush=True)
    json.dump(res, open(os.path.join(OUT, "result.json"), "w"), ensure_ascii=False, indent=1)
