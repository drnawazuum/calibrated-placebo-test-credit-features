import pandas as pd, numpy as np, sys, time
from joblib import Parallel, delayed
from sim import one_rep
R = int(sys.argv[1]) if len(sys.argv) > 1 else 300
cells = [(0.0, 0.0), (0.0, 0.5), (0.0, 0.9), (0.1, 0.0), (0.1, 0.9), (0.2, 0.0), (0.2, 0.9), (0.3, 0.0), (0.3, 0.9)]
out = []
t0 = time.time()
for g, ph in cells:
    rows = Parallel(n_jobs=2)(delayed(one_rep)(g, ph, 10000 + k) for k in range(R))
    out += rows
    pd.DataFrame(out).to_csv("sim_results.csv", index=False)
    d = pd.DataFrame(rows)
    print(f"gamma={g} phi_s={ph}: standard pass {d.std_pass.mean():.3f} | iid-null p<.05 {(d.p_iid<.05).mean():.3f} | persistence-matched p<.05 {(d.p_ar<.05).mean():.3f}  ({time.time()-t0:.0f}s)", flush=True)
print("DONE", flush=True)
