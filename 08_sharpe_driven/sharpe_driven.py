"""Sharpe-Driven Stock Selection and Liquidity-Constrained Portfolio Optimization（arXiv:2511.13251）复现。
三步：① 选股池（流动性 + 市值过滤 → 滚动夏普排名取前一档）② 权重 = 0.5×波动率倒数 + 0.5×夏普占比，每天调仓
③ 回撤控制：回撤 >2% 仓位 80%，4%~6% 仓位 40%，>6% 清仓并冷却一天。买卖各 5bp。
论文没给的参数（回看窗口、分位、流动性门槛）用 LOOK / TOPQ / MIN_AMT，可用命令行改：python3 sharpe_driven.py [LOOK] [TOPQ]
数据：01_dmfm/prices.py 生成的 prices.npz（沪深300 / 中证500 / 中证1000 历史成分股）。"""
import os, sys, json, numpy as np, pandas as pd
DATA = os.environ.get("QPR_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "08_sharpe_driven"); os.makedirs(OUT, exist_ok=True)
LOOK = int(sys.argv[1]) if len(sys.argv) > 1 else 60; TOPQ = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10
MIN_AMT = 5e7; COST = 0.0005
Z = np.load(os.path.join(DATA, "prices.npz"), allow_pickle=True); D = pd.to_datetime(Z["dates"])
t0 = int(np.searchsorted(D.values, np.datetime64("2022-01-01"))); D = D[t0:]
c = (Z["close"][t0 - 1:] * Z["factor"][t0 - 1:]).astype(np.float64); ret = c[1:] / c[:-1] - 1; ret[~np.isfinite(ret)] = np.nan
amt = Z["amount"][t0:].astype(np.float64) * 1e3                                              # 千元 → 元
mem = (Z["m_csi300"][t0:] > 0) | (Z["m_csi500"][t0:] > 0) | (Z["m_csi1000"][t0:] > 0)         # 市值过滤：只留三大指数成分股（去掉微盘）
traded = np.isfinite(Z["close"][t0:]) & (np.nan_to_num(Z["volume"][t0:]) > 0)
R = pd.DataFrame(ret); mu = R.rolling(LOOK, min_periods=LOOK).mean().values; sd = R.rolling(LOOK, min_periods=LOOK).std().values
shp = mu / sd * np.sqrt(244); liq = pd.DataFrame(amt).rolling(20, min_periods=10).mean().values
T, N = ret.shape
def target(t, mode):
    """t 日收盘后算出的目标权重"""
    ok = mem[t] & traded[t] & np.isfinite(shp[t]) & (liq[t] >= MIN_AMT) & (sd[t] > 0)
    w = np.zeros(N)
    if ok.sum() < 20: return w
    if mode == "ew_all": w[ok] = 1 / ok.sum(); return w
    k = max(5, int(ok.sum() * TOPQ)); s = np.where(ok, shp[t], -np.inf); sel = np.argsort(-s)[:k]; sel = sel[s[sel] > 0]
    if len(sel) == 0: return w
    if mode == "ew_top": w[sel] = 1 / len(sel); return w
    if mode == "mv":                                                                             # 经典均值方差（最大夏普，只做多，协方差向对角收缩 0.5）
        from scipy.optimize import minimize
        X = np.nan_to_num(ret[t - LOOK + 1:t + 1][:, sel]); S = np.cov(X.T); S = 0.5 * S + 0.5 * np.diag(np.diag(S)); m = mu[t, sel]
        f = lambda x: -(x @ m) / np.sqrt(x @ S @ x + 1e-12)
        r = minimize(f, np.ones(len(sel)) / len(sel), bounds=[(0, 1)] * len(sel), constraints={"type": "eq", "fun": lambda x: x.sum() - 1}, method="SLSQP", options={"maxiter": 60})
        w[sel] = np.clip(r.x, 0, None) / np.clip(r.x, 0, None).sum(); return w
    iv = 1 / sd[t, sel]; w[sel] = 0.5 * iv / iv.sum() + 0.5 * s[sel] / s[sel].sum(); return w   # 论文公式
def run(mode, start, end, risk=True):
    a = int(np.searchsorted(D.values, np.datetime64(start))); b = int(np.searchsorted(D.values, np.datetime64(end), side="right"))
    nav, peak, w, expo, cool = 1.0, 1.0, np.zeros(N), 1.0, 0; out = []; tov = 0.0; pend = None
    for t in range(a, b):
        r = np.nan_to_num(ret[t]); g = float(w @ r); nav *= 1 + g; w = w * (1 + r) / (1 + g)             # 持仓随价格漂移，现金部分不动
        if pend is not None:                                                                    # 上一交易日收盘后的信号，今天收盘成交
            tr = np.abs(pend - w).sum(); nav *= 1 - COST * tr; tov += tr / 2; w = pend; pend = None
        peak = max(peak, nav); dd = nav / peak - 1
        if risk:
            if cool > 0: cool -= 1; expo = 0.0; peak = nav if cool == 0 else peak               # 冷却结束后，从当前净值重新计回撤
            elif dd < -0.06: expo, cool = 0.0, 1
            elif dd < -0.04: expo = 0.4
            elif dd < -0.02: expo = 0.8
            else: expo = 1.0
        pend = target(t, mode) * expo
        out.append(nav)
    s = pd.Series(out, D[a:b]); r = s.pct_change().fillna(s.iloc[0] - 1); yrs = len(s) / 244
    return s, dict(ann=round((s.iloc[-1] ** (1 / yrs) - 1) * 100, 2), vol=round(r.std() * np.sqrt(244) * 100, 2), sharpe=round(r.mean() / r.std() * np.sqrt(244), 2), mdd=round((s / s.cummax() - 1).min() * 100, 2), turnover=round(tov / yrs, 1))
def bench(start, end):
    a = int(np.searchsorted(D.values, np.datetime64(start))); b = int(np.searchsorted(D.values, np.datetime64(end), side="right")); ix = Z["i_csi300"][t0:].astype(np.float64)
    s = pd.Series(ix[a:b] / ix[a - 1], D[a:b]); r = s.pct_change().fillna(s.iloc[0] - 1); yrs = len(s) / 244
    return s, dict(ann=round((s.iloc[-1] ** (1 / yrs) - 1) * 100, 2), vol=round(r.std() * np.sqrt(244) * 100, 2), sharpe=round(r.mean() / r.std() * np.sqrt(244), 2), mdd=round((s / s.cummax() - 1).min() * 100, 2), turnover=0)
if __name__ == "__main__":
    res = {}; curves = {}; suf = "" if (LOOK, TOPQ) == (60, 0.10) else f"_L{LOOK}_Q{TOPQ}"
    for tag, (s, e) in {"paper_2023-01_2025-08": ("2023-01-01", "2025-08-31"), "after_2025-09_2026-11": ("2025-09-01", "2026-11-16"), "all_2023-01_2026-11": ("2023-01-01", "2026-11-16")}.items():
        res[tag] = {}
        for name, (mode, risk) in {"论文方法（夏普选股+混合权重+回撤控制）": ("paper", True), "同上，不加回撤控制": ("paper", False), "选中股票等权": ("ew_top", False), "全池等权": ("ew_all", False), "均值方差": ("mv", False)}.items():
            if mode == "mv" and (tag.startswith("all") or suf): continue                           # 均值方差很慢，只在默认参数、分段区间上跑
            cv, m = run(mode, s, e, risk); res[tag][name] = m; print(tag, name, m, flush=True)
            if tag.startswith("all"): curves[name] = cv
        cv, m = bench(s, e); res[tag]["沪深300"] = m; print(tag, "沪深300", m, flush=True)
        if tag.startswith("all"): curves["沪深300"] = cv
    res["params"] = dict(lookback=LOOK, top_quantile=TOPQ, min_amount=MIN_AMT, cost_per_side=COST)
    json.dump(res, open(os.path.join(OUT, f"result{suf}.json"), "w"), ensure_ascii=False, indent=1)
    if not suf: pd.DataFrame(curves).round(4).to_csv(os.path.join(OUT, "nav.csv"))
