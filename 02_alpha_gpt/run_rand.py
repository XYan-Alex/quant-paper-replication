import os
RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
DATA = os.environ.get("QPR_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
import sys,json,time,random; sys.path.insert(0,'.'); import eng
idx=[int(i) for i in sys.argv[2].split(',')]; out={}
for i in idx:
    random.seed(1000+i); seeds=[eng.rnd_tree(2) for _ in range(3)]
    P,h=eng.search(seeds,gens=10,pop=20,seed=1000+i); top=[(eng.tostr(p),eng.fitness(p)) for p in P[:5]]
    out[i]=dict(name='random',seeds=[eng.tostr(s) for s in seeds],seed_fit=[eng.fitness(s) for s in seeds],hist=h,top=top)
    print(i,round(h[0]['is_ic'],4),'→',round(h[-1]['is_ic'],4),'os',round(h[-1]['os_ic'],4),h[-1]['best'][:80],flush=True)
    json.dump(out,open(os.path.join(RUNS,f'ag_{sys.argv[1]}.json'),'w'))
print('done',flush=True)
