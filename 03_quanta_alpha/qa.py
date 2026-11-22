"""QuantaAlpha 精简复现：假设→公式的“轨迹”，三轮（多样化初始化 / 定点变异 / 交叉），复杂度+冗余闸门，
因子池 → LightGBM → 沪深300 测试集（2022-01 ~ 2025-12），并迁移到中证500"""
import os
DATA = os.environ.get("QPR_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
import sys,json,numpy as np,bottleneck as bn,lightgbm as lgb; sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','02_alpha_gpt')); import eng
from eng import Z,t0,cols,D,T,N,F
from ideas import IDEAS
R0=[s for _,_,s in IDEAS]
R1=[s for v in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','02_alpha_gpt','it_seeds.json'))).values() for s in v]
R2=["mul(cs_rank(neg(ts_std(ret,20))),cs_rank(neg(ts_corr(close,volume,10))))","neg(mul(ts_std(ret,20),div(ts_mean(volume,5),ts_mean(volume,60))))",
"neg(ts_corr(absv(ret),volume,20))","neg(ts_mean(mul(div(sub(high,close),close),div(volume,ts_mean(volume,20))),10))",
"neg(ts_max(mul(ret,div(volume,ts_mean(volume,20))),20))","neg(ts_corr(sub(div(open,shift(close,1)),1),shift(ret,1),20))",
"neg(ts_mean(mul(sub(div(open,shift(close,1)),1),div(volume,ts_mean(volume,20))),10))","neg(div(ts_std(ret,5),ts_std(ret,60)))",
"neg(div(ts_mean(div(sub(high,low),close),5),ts_mean(div(sub(high,low),close),60)))","neg(mul(ts_delta_ratio(close,20),ts_std(ret,20)))",
"neg(ts_corr(ret,ts_delta_ratio(volume,1),20))","neg(mul(ts_mean(div(sub(close,vwap),vwap),5),ts_std(ret,20)))","neg(ts_rank(ts_std(ret,20),120))",
"neg(ts_rank(div(ts_mean(amount,5),ts_mean(amount,60)),60))","neg(ts_maxmin_scale(ts_mean(amount,5),120))","neg(sub(ts_max(ret,20),absv(ts_min(ret,20))))",
"neg(ts_mean(sign(sub(close,open)),20))","neg(ts_decay(mul(div(sub(close,open),open),div(volume,ts_mean(volume,20))),10))",
"neg(ts_corr(ts_delta_ratio(close,1),ts_delta_ratio(volume,1),10))","neg(ts_mean(div(amount,ts_mean(amount,120)),20))","neg(div(ts_max(high,20),ts_min(low,20)))",
"neg(mul(ts_maxmin_scale(close,60),div(ts_mean(volume,5),ts_mean(volume,60))))","neg(ts_std(div(volume,ts_mean(volume,20)),20))","neg(ts_mean(absv(sub(div(close,vwap),1)),20))"]
def subtrees_set(t):
    out=set()
    def rec(n):
        if isinstance(n,tuple):
            for a in n[1:]: rec(a)
            out.add((eng.tostr(n),eng.size(n)))
    rec(t); return out
def lcs(a,b):
    A=subtrees_set(a);B={s for s,_ in subtrees_set(b)}; return max([sz for s,sz in A if s in B],default=0)
def gate(exprs,zoo,max_size=18,max_sim=0.7):
    keep=[];rej=[]
    for s in exprs:
        t=eng.parse(s); sz=eng.size(t)
        sim=max([lcs(t,eng.parse(z))/sz for z in zoo+keep],default=0)
        if sz>max_size: rej.append((s,'complex'))
        elif sim>max_sim: rej.append((s,'redundant'))
        else: keep.append(s)
    return keep,rej
def feats(exprs,M):
    cu=M.any(0); X=[]
    for s in exprs:
        x=np.where(M,eng.ev(eng.parse(s)),np.nan); r=bn.nanrankdata(x,axis=1); n=np.sum(np.isfinite(r),1,keepdims=True); X.append(((r-0.5)/np.maximum(n,1)).astype(np.float32)[:,cu])
    return np.stack(X,-1)
c=F['close']; lab=np.full((T,N),np.nan,np.float32); lab[:T-2]=c[2:]/c[1:T-1]-1
def fit_eval(exprs,univ='m_csi300',idx='i_csi300',model=None,seed=0):
    M=Z[univ][t0:][:,cols]>0; X=feats(exprs,M); cu=M.any(0); y=np.where(M,lab,np.nan)[:,cu]; M=M[:,cu]; Nn=int(cu.sum()); cc=c[:,cu]
    yz=(y-np.nanmean(y,1,keepdims=True))/(np.nanstd(y,1,keepdims=True)+1e-9)
    def sl(a,b):
        d=(D>=np.datetime64(a))&(D<=np.datetime64(b)); m=d[:,None]&np.isfinite(yz); return X[m],yz[m],d
    if model is None:
        Xtr,ytr,_=sl('2016-01-01','2020-12-31'); Xva,yva,_=sl('2021-01-01','2021-12-31')
        model=lgb.LGBMRegressor(n_estimators=600,learning_rate=0.05,num_leaves=31,max_depth=6,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,min_child_samples=200,reg_lambda=10,n_jobs=2,verbose=-1,random_state=seed)
        model.fit(Xtr,ytr,eval_set=[(Xva,yva)],callbacks=[lgb.early_stopping(50,verbose=False)])
    d=(D>=np.datetime64('2022-01-01'))&(D<=np.datetime64('2025-12-26')); days=np.where(d)[0]
    pred=np.full((T,Nn),np.nan,np.float32)
    for i in days:
        m=M[i]&np.isfinite(X[i]).any(1)
        if m.sum()>50: pred[i,m]=model.predict(X[i][m])
    ic=[];ric=[]
    for i in days:
        m=np.isfinite(pred[i])&np.isfinite(y[i])
        if m.sum()<50: continue
        ic.append(np.corrcoef(pred[i][m],y[i][m])[0,1]); ric.append(np.corrcoef(bn.rankdata(pred[i][m]),bn.rankdata(y[i][m]))[0,1])
    ic=np.array(ic);ric=np.array(ric)
    # TopkDropout(50,5)，信号日次日收盘成交，买0.05% 卖0.15%
    ix=Z[idx][t0:]; hold=[]; ex=[]; pr=[]
    r1=cc[1:]/cc[:-1]-1
    for i in days[:-2]:
        s=pred[i]; order=[j for j in np.argsort(-np.nan_to_num(s,nan=-9)) if np.isfinite(s[j])]
        cost=0
        if not hold: hold=order[:50]; cost=0.0005
        else:
            rk={j:k for k,j in enumerate(order)}; held=sorted(hold,key=lambda j:rk.get(j,9999)); cand=[j for j in order if j not in hold][:5]
            comb=sorted(held+cand,key=lambda j:rk.get(j,9999)); drop=[j for j in comb[-5:] if j in hold]
            buy=cand[:len(drop)]; hold=[j for j in hold if j not in drop]+buy; cost=len(drop)/50*(0.0005+0.0015)
        rr=np.nan_to_num(r1[i+1][hold]).mean()-cost      # i+1收盘 → i+2收盘
        pr.append(rr); ex.append(rr-(ix[i+2]/ix[i+1]-1))
    ex=np.array(ex); cum=np.cumsum(ex); pr=np.array(pr)
    res=dict(n=len(exprs),IC=round(float(ic.mean()),4),ICIR=round(float(ic.mean()/ic.std()),3),RankIC=round(float(ric.mean()),4),RankICIR=round(float(ric.mean()/ric.std()),3),
             ARR=round(float(ex.mean()*252*100),2),IR=round(float(ex.mean()/ex.std()*np.sqrt(252)),2),MDD=round(float((cum-np.maximum.accumulate(cum)).min()*100),2),
             cum_excess=round(float(cum[-1]*100),2),port_ann=round(float(pr.mean()*252*100),2),best_iter=int(getattr(model,'best_iteration_',0) or 0))
    return res,model,cum
if __name__=='__main__':
    out={}
    p0,_=gate(R0,[]); p1,rj1=gate(R1,p0); p2,rj2=gate(R2,p0+p1)
    print('gate: R0',len(p0),'R1',len(p1),'rej',len(rj1),'R2',len(p2),'rej',len(rj2),flush=True)
    for nm,ex in [('R0 初始化',p0),('R0+变异',p0+p1),('R0+变异+交叉',p0+p1+p2)]:
        rs=[]
        for sd in range(3):
            r,m,cum=fit_eval(ex,seed=sd); rs.append(r)
            if sd==0: 
                r5,_,cum5=fit_eval(ex,'m_csi500','i_csi500',model=m); 
        avg={k:round(float(np.mean([x[k] for x in rs])),4) for k in rs[0]}
        out[nm]=dict(csi300=avg,csi500_transfer=r5); print(nm,'CSI300',avg,flush=True); print('   → CSI500 迁移',r5,flush=True)
        if nm.endswith('交叉'): np.save(os.path.join(DATA,'qa_cum300.npy'),cum); np.save(os.path.join(DATA,'qa_cum500.npy'),cum5)
    json.dump(out,open(os.path.join(DATA,'qa_res.json'),'w'),ensure_ascii=False)
