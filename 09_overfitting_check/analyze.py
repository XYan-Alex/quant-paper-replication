import os
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "results", "09_overfitting_check"); os.makedirs(OUT, exist_ok=True)
import os, sys, numpy as np, pandas as pd; sys.path.insert(0,HERE); from stats import *

def rep(name, f, pick, sel, hold=None, S=16):
    df=pd.read_csv(f,index_col=0,parse_dates=True)
    A=df.loc[sel[0]:sel[1]]; srs=np.array([sr(A[c]) for c in A])
    p,_=pbo(A.values,S=S); pf,_=pbo(df.values,S=16)
    rank=int((srs>sr(A[pick])).sum())+1
    print(f'== {name}  配置数 {df.shape[1]}  选中 {pick}（选择区间内 Sharpe 排第 {rank}）')
    print(f'  PBO 选择区间 {sel[0]}~{sel[1]} ({len(A)} 月, S={S}) = {p:.2f} ；全区间 = {pf:.2f}')
    for N in (df.shape[1], 3*df.shape[1], 100):
        d,s,s0,T=dsr(A[pick],srs,N); print(f'  DSR N={N:3d}: {d:.3f}  (月度 SR {s:.3f} = 年化 {s*np.sqrt(12):.2f}，门槛 SR0 {s0:.3f} = 年化 {s0*np.sqrt(12):.2f}，T={T})')
    if hold:
        H=df.loc[hold[0]:hold[1]][pick]; d,s,_,T=dsr(H,[0,0],1); print(f'  留出区间 {hold[0]}~{hold[1]}：月均超额 {H.mean()*100:.2f}%，年化 IR {s*np.sqrt(12):.2f}，P(SR>0)={d:.3f}，T={T}')
for u in ['csi1000','csi500','csi300']:
    try: rep(f'DMFM {u}',os.path.join(OUT,f'dmfm_{u}.csv'),'ours',('2016-01','2022-06'),('2022-07','2026-11'))
    except Exception as e: print(u,'skip',e)
for u in ['m_csi1000','m_csi500','m_csi300']:
    rep(f'AlphaGPT {u}',os.path.join(OUT,f'ag_{u}.csv'),'h20se_h20it|M4',('2022-01','2026-11'),None,S=8)
    df=pd.read_csv(os.path.join(OUT,f'ag_{u}.csv'),index_col=0,parse_dates=True); print('   各配置样本外年化IR:',{c:round(sr(df.loc['2022-01':][c])*np.sqrt(12),2) for c in df})
