import os
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "results", "09_overfitting_check"); os.makedirs(OUT, exist_ok=True)
DATA = os.environ.get("QPR_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0,os.path.join(HERE,'..','01_dmfm')); sys.path.insert(0,HERE)
from backtest import *; from stats import *
def z(S): return (S-np.nanmean(S,1,keepdims=True))/(np.nanstd(S,1,keepdims=True)+1e-9)
out={}
for u in sys.argv[1:]:
    X,Y,idx,dates,groups,names=panel(u); bench=P["i_"+u][t0:].astype(np.float64)
    SC={"ew":ew_score(X,groups).astype(np.float64),"linear":linear_score(X,Y,idx,dates)}; SC["ew"][idx<0]=np.nan
    for k in ("mlp","mgat","dmfm","dmfmf","lite","lite2","ours"):
        f=f"{D}/score_{u}_{k}.npy"
        if os.path.exists(f): SC[k]=np.load(f)
    for name,ks in {"ew+dmfm":["ew","dmfm"],"ew+linear":["ew","linear"],"ew+dmfm+lite":["ew","dmfm","lite"],"ew+dmfm+linear":["ew","dmfm","linear"],"all4":["ew","dmfm","lite","linear"]}.items():
        if all(k in SC for k in ks): SC[name]=sum(z(SC[k]) for k in ks)
    R={}
    for k,S in SC.items():
        nv,bn,_=backtest(S,idx,dates,bench,start="2016-01-01"); R[k]=monthly(nv.values,bn.values,nv.index)
    df=pd.DataFrame(R).dropna(); df.to_csv(os.path.join(OUT,f'dmfm_{u}.csv')); print(u,df.shape,list(df.columns),flush=True)
