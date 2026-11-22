"""把搜索得到的因子去冗余、合成，在中证1000上做月频前10%回测（样本外 2022-01 起）"""
import os
RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
DATA = os.environ.get("QPR_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
import sys,json,numpy as np,bottleneck as bn; sys.path.insert(0,'.'); import eng
from eng import Z,t0,cols,D,T,N
def zs(x,m):
    x=np.where(m,x,np.nan); r=bn.nanrankdata(x,axis=1); n=np.sum(np.isfinite(r),1,keepdims=True); return (r-0.5)/np.maximum(n,1)-0.5
def load_pool(tags,per=3):
    cand=[]
    if tags==['seed']:                                # 只用人写的 16 个种子公式（不搜索），IC 取 20 日目标的那次运行
        for part in 'ab':
            d=json.load(open(os.path.join(RUNS,f'ag_h20se_{part}.json')))
            for i,v in d.items(): cand.append((v['seeds'][0],v['seed_fit'][0],int(i),'seed'))
        return cand
    for tag in tags:
        for part in 'ab':
            d=json.load(open(os.path.join(RUNS,f'ag_{tag}_{part}.json')))
            for i,v in d.items():
                for s,f in v['top'][:per]: cand.append((s,f,int(i),tag))
    return cand
def select(cand,maxcorr=0.7,minic=0.03):
    cand=sorted(cand,key=lambda c:-abs(c[1][0])); keep=[]; X=[]
    days=np.where(eng.IS)[0][::20]
    for s,f,i,tag in cand:
        if abs(f[0])<minic: continue
        x=eng.ev(eng.parse(s))*np.sign(f[0]); r=zs(x,eng.mem)[days]
        ok=True
        for y in X:
            m=np.isfinite(r)&np.isfinite(y)
            if abs(np.corrcoef(r[m],y[m])[0,1])>maxcorr: ok=False;break
        if ok: keep.append((s,f,i,tag)); X.append(r)
    return keep
def combine(keep):
    S=np.zeros((T,N),np.float32); C=np.zeros((T,N),np.float32)
    for s,f,i,tag in keep:
        r=zs(eng.ev(eng.parse(s))*np.sign(f[0]),eng.mem); m=np.isfinite(r); S[m]+=r[m]; C[m]+=1
    return np.where(C>=max(1,len(keep)//2),S/np.maximum(C,1),np.nan)
def backtest(score,univ='m_csi1000',idx='i_csi1000',start='2022-01-01',end='2026-11-16',top=0.10,cost=0.004,freq='M'):
    M=Z[univ][t0:][:,cols]>0; c=eng.F['close']; ix=Z[idx][t0:]
    days=np.where((D>=np.datetime64(start))&(D<=np.datetime64(end)))[0]
    mon=D.astype('datetime64[M]'); nav=[1.0]; w=np.zeros(N); pend=None; tov=0
    for k,i in enumerate(days[1:],1):
        r=c[i]/c[i-1]-1; r[~np.isfinite(r)]=0; g=float(w@r); nav.append(nav[-1]*(1+g))
        if w.sum()>0: w=w*(1+r); w/=max(w.sum(),1e-9) 
        if pend is not None:                         # 信号次日收盘换仓
            t=np.abs(pend-w).sum(); nav[-1]*=1-cost*t; tov+=t/2; w=pend; pend=None
        if i+1<T and ((mon[i+1]!=mon[i]) if freq=='M' else (k%freq==0)):               # 月末出信号
            s=np.where(M[i]&np.isfinite(c[i]),score[i],np.nan); n=int(np.isfinite(s).sum()*top)
            if n>5:
                sel=np.argsort(-np.nan_to_num(s,nan=-9))[:n]; pend=np.zeros(N); pend[sel]=1/n
    nav=np.array(nav); b=ix[days]/ix[days[0]]; yrs=len(days)/244
    rr=nav[1:]/nav[:-1]-1; ex=nav/b
    return dict(ann=round((nav[-1]**(1/yrs)-1)*100,2),bench=round((b[-1]**(1/yrs)-1)*100,2),excess=round(((nav[-1]/b[-1])**(1/yrs)-1)*100,2),mdd=round((nav/np.maximum.accumulate(nav)-1).min()*100,1),
                ex_mdd=round(float(np.nanmin(ex/np.maximum.accumulate(np.nan_to_num(ex,nan=0))-1))*100,1),sharpe=round(rr.mean()/rr.std()*np.sqrt(244),2),tov=round(tov/yrs,1),cum=round(nav[-1],3),bcum=round(b[-1],3)),nav,b,D[days]
if __name__=='__main__':
    tags=sys.argv[1].split(','); cand=load_pool(tags); keep=select(cand,minic=0.0 if tags==['seed'] else 0.03); print('pool',len(cand),'→',len(keep))
    for s,f,i,tag in keep: print(' ',tag,i,round(f[0],4),round(f[2],4),s[:110])
    sc=combine(keep); np.save(f'{DATA}/ag_score_{"_".join(tags)}.npy',sc)
    a=eng.ic_series(sc,eng.IS); b=eng.ic_series(sc,eng.OS); print('combo IC IS',round(a.mean(),4),'OS',round(b.mean(),4),'ICIR OS',round(b.mean()/b.std(),2))
    for u,ix in [('m_csi1000','i_csi1000'),('m_csi500','i_csi500'),('m_csi300','i_csi300')]:
        for st,en in [('2016-01-01','2021-12-31'),('2022-01-01','2026-11-16')]:
            print(u,st[:4],backtest(sc,u,ix,st,en)[0])
    json.dump([dict(expr=s,is_ic=f[0],os_ic=f[2],idea=i,tag=tag) for s,f,i,tag in keep],open(f'{DATA}/ag_pool_{"_".join(tags)}.json','w'),ensure_ascii=False)
