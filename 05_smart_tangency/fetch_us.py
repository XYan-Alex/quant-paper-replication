"""下载美股行业 ETF、道指成分股、SPY / DIA 的复权日线（Yahoo Finance），存到 data/us/。"""
import os, sys, time, json, requests, pandas as pd
DATA = os.environ.get("QPR_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
OUT = os.path.join(DATA, "us"); os.makedirs(OUT, exist_ok=True)
ETF = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]                         # 9 只 SPDR 行业 ETF（论文用 12 只，未列全名单；这 9 只自 1998 年起有完整历史）
DJIA = ["AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT", "DOW"]
def get(sym):
    u = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1=883612800&period2=1795000000&interval=1d&events=div%2Csplit"
    for k in range(4):
        try:
            j = requests.get(u, timeout=30, headers={"User-Agent": "Mozilla/5.0"}).json()["chart"]["result"][0]
            return pd.DataFrame({"date": pd.to_datetime(j["timestamp"], unit="s").normalize(), "adj": j["indicators"]["adjclose"][0]["adjclose"], "close": j["indicators"]["quote"][0]["close"], "volume": j["indicators"]["quote"][0]["volume"]}).dropna(subset=["adj"])
        except Exception as e: err = e; time.sleep(2)
    raise err
if __name__ == "__main__":
    for s in ETF + ["SPY", "DIA"] + DJIA:
        p = os.path.join(OUT, s + ".parquet")
        if os.path.exists(p): continue
        try: d = get(s); d.to_parquet(p); print(s, len(d), str(d.date.iloc[0])[:10], str(d.date.iloc[-1])[:10], flush=True)
        except Exception as e: print(s, "ERR", repr(e)[:100], flush=True)
        time.sleep(0.4)
