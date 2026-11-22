"""预先算好每个交易日、每个风险度量（方差 / 半方差 / CVaR）、每档 λ 的有效前沿权重。
λ ∈ {5,10,…,100}：目标收益 = 最低资产均值 + λ% ×（最高 − 最低）；在此约束下最小化风险，只做多、权重和为 1。回看 252 天。
用法：python3 frontier.py etf|djia [mv,sv,cvar]"""
import os, sys, numpy as np, pandas as pd, quadprog, time
from scipy.optimize import linprog
DATA = os.environ.get("QPR_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
from fetch_us import ETF, DJIA
LOOK = 252; LAMS = np.arange(5, 101, 5)
def load(u):
    syms = ETF if u == "etf" else [s for s in DJIA if s not in ("WBA", "DOW", "V", "CRM")]       # 道指：去掉历史不全的
    P = pd.DataFrame({s: pd.read_parquet(os.path.join(DATA, "us", s + ".parquet")).set_index("date")["adj"] for s in syms + ["SPY"]}).dropna()
    P = P[P.index >= ("2001-01-01" if u == "etf" else "2003-01-01")]; return P[syms], P["SPY"]
def qp(S, mu, m0):
    n = len(mu); S = S + 1e-8 * np.eye(n) * np.trace(S) / n
    C = np.column_stack([np.ones(n), mu, np.eye(n)]); b = np.concatenate([[1.0, m0 - 1e-12], np.zeros(n)])
    try: w = quadprog.solve_qp(2 * S, np.zeros(n), C, b, 1)[0]
    except ValueError: w = np.zeros(n); w[np.argmax(mu)] = 1
    w = np.clip(w, 0, None); return w / w.sum()
def cvar(Y, mu, m0, beta=0.95):
    q, n = Y.shape; c = np.concatenate([np.zeros(n), [1.0], np.ones(q) / (q * (1 - beta))])     # 变量：w, l, S_k
    A = np.hstack([-Y, -np.ones((q, 1)), -np.eye(q)]); A = np.vstack([A, np.concatenate([-mu, [0], np.zeros(q)])]); b = np.concatenate([np.zeros(q), [-m0 + 1e-12]])
    r = linprog(c, A_ub=A, b_ub=b, A_eq=np.concatenate([np.ones(n), [0], np.zeros(q)])[None], b_eq=[1], bounds=[(0, 1)] * n + [(None, None)] + [(0, None)] * q, method="highs")
    if r.status != 0: w = np.zeros(n); w[np.argmax(mu)] = 1; return w
    w = np.clip(r.x[:n], 0, None); return w / w.sum()
if __name__ == "__main__":
    u = sys.argv[1]; kinds = (sys.argv[2] if len(sys.argv) > 2 else "mv,sv,cvar").split(",")
    P, _ = load(u); Y = np.log(P).diff().dropna().values; T, n = Y.shape
    for kind in kinds:
        W = np.full((T, len(LAMS), n), np.nan, np.float32); t0 = time.time()
        for t in range(LOOK, T):
            y = Y[t - LOOK + 1:t + 1]; mu = y.mean(0); lo, hi = mu.min(), mu.max()
            if kind == "mv": S = np.cov(y.T)
            elif kind == "sv": d = np.minimum(y - mu, 0); S = d.T @ d / len(y)                    # 半协方差，基准 = 均值
            for j, lam in enumerate(LAMS):
                m0 = lo + lam / 100 * (hi - lo)
                W[t, j] = cvar(y, mu, m0) if kind == "cvar" else qp(S, mu, m0)
            if t % 1000 == 0: print(u, kind, t, T, round(time.time() - t0), flush=True)
        np.save(os.path.join(DATA, "us", f"front_{u}_{kind}.npy"), W); print(u, kind, "done", round(time.time() - t0), flush=True)
