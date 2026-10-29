import pandas as pd, sys, time
from joblib import Parallel, delayed
from sim2 import one_rep
R = int(sys.argv[1]) if len(sys.argv) > 1 else 300
# (gamma, phi_s, phi_c, sd_c, fe)
cells = [(0.0, 0.9, 0.9, 0.5, False),   # no period control at all
         (0.0, 0.0, 0.9, 0.5, False),
         (0.0, 0.9, 0.99, 1.0, True),   # strongly drifting unobserved cycle, year effects only
         (0.0, 0.0, 0.99, 1.0, True),
         (0.0, 0.9, 0.99, 1.0, False)]
out = []; t0 = time.time()
for g, ph, pc, sc, fe in cells:
    rows = Parallel(n_jobs=2)(delayed(one_rep)(g, ph, 20000 + k, 99, 100, 40, 600, pc, sc, fe) for k in range(R))
    out += rows; pd.DataFrame(out).to_csv("sim2_results.csv", index=False)
    d = pd.DataFrame(rows)
    print(f"gamma={g} phi_s={ph} phi_c={pc} sd_c={sc} year_effects={fe}: standard pass {d.std_pass.mean():.3f} | iid-null {(d.p_iid<.05).mean():.3f} | persistence-matched {(d.p_ar<.05).mean():.3f} ({time.time()-t0:.0f}s)", flush=True)
print("DONE", flush=True)
