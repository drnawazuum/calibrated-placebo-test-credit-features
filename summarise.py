"""Turns the raw simulation results into pass rates with 95% Wilson intervals -> sim_summary.csv."""
import pandas as pd, numpy as np
def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h
a = pd.read_csv("sim_results.csv"); a["phi_c"] = 0.9; a["sd_c"] = 0.5; a["fe"] = 1
b = pd.read_csv("sim2_results.csv")
d = pd.concat([a, b], ignore_index=True)
rows = []
for key, g in d.groupby(["gamma", "phi_s", "phi_c", "sd_c", "fe"]):
    n = len(g); r = dict(zip(["gamma", "phi_s", "phi_c", "sd_c", "fe"], key)); r["n"] = n
    for name, k in [("std", g.std_pass.sum()), ("iid", (g.p_iid < 0.05).sum()), ("ar", (g.p_ar < 0.05).sum())]:
        lo, hi = wilson(k, n); r[name] = k / n; r[name + "_lo"] = lo; r[name + "_hi"] = hi
    r["mean_phi_hat"] = g.phi_hat.mean(); rows.append(r)
pd.DataFrame(rows).to_csv("sim_summary.csv", index=False); print("sim_summary.csv written")
