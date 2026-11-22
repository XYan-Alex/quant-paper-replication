"""下载 30 分钟 K 线（新浪，公开接口最多约 1970 根 ≈ 一年）：沪深300 指数及其 ETF，另加中证500、上证指数、深证成指做稳健性。"""
import os, akshare as ak, pandas as pd, time
DATA = os.environ.get("QPR_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
OUT = os.path.join(DATA, "m30"); os.makedirs(OUT, exist_ok=True)
SYMS = {"sh000300": "沪深300指数", "sh510300": "沪深300ETF", "sh000905": "中证500指数", "sh510500": "中证500ETF", "sh000001": "上证指数", "sz399001": "深证成指"}
if __name__ == "__main__":
    for s, n in SYMS.items():
        for k in range(3):
            try:
                d = ak.stock_zh_a_hist_min_sina(symbol=s, period="30"); d["day"] = pd.to_datetime(d["day"])
                for c in ("open", "high", "low", "close", "volume"): d[c] = d[c].astype(float)
                d.to_parquet(os.path.join(OUT, s + ".parquet")); print(s, n, len(d), d.day.iloc[0], d.day.iloc[-1], flush=True); break
            except Exception as e: print(s, "ERR", repr(e)[:100]); time.sleep(3)
        time.sleep(1)
