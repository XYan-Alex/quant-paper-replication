"""抓取全市场季度业绩报表（含公告日，做 point-in-time）与资产负债表、申万行业历史。"""
import os
DATA = os.environ.get("QPR_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
import akshare as ak, pandas as pd, sys, time, os
OUT = os.path.join(DATA, "fund"); os.makedirs(OUT, exist_ok=True)
qs = [f"{y}{md}" for y in range(2008, 2027) for md in ("0331", "0630", "0930", "1231")]
qs = [q for q in qs if "20081231" <= q <= "20260930"]
for name, fn in (("yjbb", ak.stock_yjbb_em), ("zcfz", ak.stock_zcfz_em)):
    for q in qs:
        p = f"{OUT}/{name}_{q}.parquet"
        if os.path.exists(p): continue
        for k in range(4):
            try:
                df = fn(date=q); df.astype(str).to_parquet(p); print(name, q, len(df), flush=True); break
            except Exception as e:
                print("ERR", name, q, repr(e)[:120], flush=True); time.sleep(3)
try:
    sw = ak.stock_industry_clf_hist_sw(); sw.astype(str).to_parquet(f"{OUT}/sw_hist.parquet"); print("sw", len(sw), list(sw.columns))
except Exception as e: print("ERR sw", repr(e)[:200])
print("DONE")
