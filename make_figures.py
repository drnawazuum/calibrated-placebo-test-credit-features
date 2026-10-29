"""Makes Figure 1 and Figure 2 from sim_summary.csv (output of summarise.py)."""
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
t = pd.read_csv("sim_summary.csv")
BL, OR, GR = "#0072B2", "#D55E00", "#009E73"
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
sc = [("Year effects,\nmoderate cycle,\niid series", (0.0, 0.0, 0.9, 1)), ("Year effects,\nmoderate cycle,\npersistent series", (0.0, 0.9, 0.9, 1)),
      ("Year effects,\ndrifting cycle,\npersistent series", (0.0, 0.9, 0.99, 1)), ("No period control,\nmoderate cycle,\npersistent series", (0.0, 0.9, 0.9, 0)),
      ("No period control,\ndrifting cycle,\npersistent series", (0.0, 0.9, 0.99, 0))]
fig, ax = plt.subplots(figsize=(7.2, 3.5)); w = 0.26
for j, (col, lab, c) in enumerate([("std", "Standard check (bootstrap)", OR), ("iid", "Placebo test, shuffled series", GR), ("ar", "Placebo test, persistence-matched", BL)]):
    vals, lo, hi = [], [], []
    for _, key in sc:
        r = t[(t.gamma == key[0]) & (t.phi_s == key[1]) & (t.phi_c == key[2]) & (t.fe == key[3])].iloc[0]
        vals.append(100 * r[col]); lo.append(100 * (r[col] - r[col + "_lo"])); hi.append(100 * (r[col + "_hi"] - r[col]))
    ax.bar([i + (j - 1) * w for i in range(len(sc))], vals, w, color=c, label=lab, yerr=[lo, hi], error_kw=dict(lw=0.8, capsize=2, ecolor="#444"))
ax.axhline(5, color="#555", lw=0.8, ls="--"); ax.text(2.5, 6.5, "intended 5%", ha="center", fontsize=8, color="#555")
ax.set_xticks(range(len(sc))); ax.set_xticklabels([s[0] for s in sc], fontsize=7.5)
ax.set_ylabel("Share of useless series that pass (%)"); ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 1.0)); ax.set_ylim(0, 80)
plt.tight_layout(); plt.savefig("fig1.png", dpi=200); plt.close()
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
for ax, ps, tt in zip(axs, [0.0, 0.9], ["Non-persistent series", "Persistent series (phi = 0.9)"]):
    for col, lab, c in [("std", "Standard check", OR), ("iid", "Placebo, shuffled", GR), ("ar", "Placebo, persistence-matched", BL)]:
        d = t[(t.phi_s == ps) & (t.phi_c == 0.9) & (t.fe == 1)].sort_values("gamma")
        ax.plot(d.gamma, 100 * d[col], marker="o", color=c, lw=1.6, label=lab)
    ax.set_title(tt, fontsize=9); ax.set_xlabel("Size of the true effect"); ax.set_xticks([0, .1, .2, .3])
axs[0].set_ylabel("Share of real effects detected (%)"); axs[0].legend(frameon=False, fontsize=7.5, loc="upper left")
plt.tight_layout(); plt.savefig("fig2.png", dpi=200); plt.close()
print("figures ok")
