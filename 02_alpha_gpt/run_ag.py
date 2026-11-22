import os
RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
DATA = os.environ.get("QPR_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
import sys,json,time; sys.path.insert(0,'.'); import eng
from ideas import IDEAS
tag=sys.argv[1]; idx=[int(i) for i in sys.argv[2].split(',')]; gens=int(sys.argv[3]) if len(sys.argv)>3 else 10
seedfile=sys.argv[4] if len(sys.argv)>4 else None
SE=json.load(open(seedfile)) if seedfile else None
out={}
for i in idx:
    name,desc,seed=IDEAS[i]; seeds=[seed] if SE is None else SE[str(i)]
    t=time.time(); f0=[eng.fitness(eng.parse(s)) for s in seeds]
    P,h=eng.search(seeds,gens=gens,pop=20,seed=i)
    top=[(eng.tostr(p),eng.fitness(p)) for p in P[:5]]
    out[i]=dict(name=name,desc=desc,seeds=seeds,seed_fit=f0,hist=h,top=top)
    print(i,name,'seed',round(abs(f0[0][0]),4),'→',round(h[-1]['is_ic'],4),'os',round(h[-1]['os_ic'],4),h[-1]['best'],round(time.time()-t),flush=True)
    json.dump(out,open(os.path.join(RUNS,f'ag_{tag}.json'),'w'),ensure_ascii=False)
print('done',flush=True)
