import os
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "results", "09_overfitting_check"); os.makedirs(OUT, exist_ok=True)
DATA = os.environ.get("QPR_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
import sys, os, numpy as np, pandas as pd
sys.path.insert(0,os.path.join(HERE,'..','02_alpha_gpt')); sys.path.insert(0,HERE)
import pool_bt as pb; from stats import monthly
sc={k:np.load(f'{DATA}/ag_score_{k}.npy') for k in ['seed','rnd','se_it','h20se_h20it']}
for u,ix in [('m_csi1000','i_csi1000'),('m_csi500','i_csi500'),('m_csi300','i_csi300')]:
    R={}
    for k,S in sc.items():
        for tag,fr,c in [('M4',  'M',0.004),('W2',10,0.0015)]:
            r,nav,b,d=pb.backtest(S,u,ix,'2016-01-01','2026-11-16',cost=c,freq=fr)
            R[f'{k}|{tag}']=monthly(nav,b*1.0,pd.to_datetime(d))
    df=pd.DataFrame(R).dropna(); df.to_csv(os.path.join(OUT,f'ag_{u}.csv')); print(u,df.shape,flush=True)
