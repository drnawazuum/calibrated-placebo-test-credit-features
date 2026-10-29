"""Real-data confirmation: apply the persistence-matched placebo test to the REAL tone series on Lending Club/FOMC and Bondora/ECB.
Usage: python3 real.py {lc|bondora} {none|year|quarter} K NSUB OUT"""
import sys, os, time
import numpy as np, pandas as pd
os.environ["OMP_NUM_THREADS"] = "1"
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
R = os.environ.get("PAPER1_DIR", "rq1_real_test")  # folder with the data and scripts of the companion paper (SSRN 7622180); sys.path.insert(0, R); os.chdir(R)
which, ctl, K, NSUB, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
if which == "lc":
    import run_pd_sentiment_test as lc
    df = lc.load_lending_club("data/loan.csv"); feats = [c for c in lc.BASELINE_FEATURES if c in df.columns]
    tone = pd.read_csv("fomc_sentiment_scores.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    ycol, dcol = "default", "issue_date"
else:
    import run_bondora_eu_test as B
    df, _ = B.load("bondora/LoanData.csv"); ctry = [c for c in df.columns if c.startswith("ctry_")]; feats = B.NUMERIC + ctry
    tone = pd.read_csv("ecb_sentiment_scores_2009_2026.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    ycol, dcol = "default12", "LoanDate"
tone["sid"] = np.arange(len(tone))
m = pd.merge_asof(df.sort_values(dcol), tone[["date", "sid"]], left_on=dcol, right_on="date", direction="backward")
m = m.dropna(subset=(feats if which == "lc" else []) + [ycol, "sid"]).reset_index(drop=True)
if len(m) > NSUB: m = m.sample(NSUB, random_state=1).reset_index(drop=True)
sid = m["sid"].astype(int).values; y = m[ycol].values
per = m[dcol].dt.year.astype(str).values if ctl == "year" else (m[dcol].dt.to_period("Q").astype(str).values if ctl == "quarter" else None)
Xb = m[feats].astype(float).values
if per is not None: Xb = np.hstack([Xb, pd.get_dummies(pd.Series(per), dtype=float).values])
tr, te = train_test_split(np.arange(len(m)), test_size=0.25, random_state=42, stratify=y); yt = y[te]
gbm = lambda: HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=42)
real = tone["net_sentiment"].values; n = len(real); mu, sd = real.mean(), real.std(); phi = np.corrcoef(real[:-1], real[1:])[0, 1]
def feat(s):
    v = s[sid]
    if per is not None: v = v - pd.Series(v).groupby(per).transform("mean").values
    return v.reshape(-1, 1)
def ar1(rng):
    e = rng.standard_normal(n); z = np.zeros(n); z[0] = e[0]
    for t in range(1, n): z[t] = phi * z[t - 1] + e[t] * np.sqrt(1 - phi ** 2)
    return mu + sd * (z - z.mean()) / z.std()
p0 = gbm().fit(Xb[tr], y[tr]).predict_proba(Xb[te])[:, 1]; a0 = roc_auc_score(yt, p0)
def gain(s):
    X = np.hstack([Xb, feat(s)]); p = gbm().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]; return roc_auc_score(yt, p) - a0, p
g_real, p1 = gain(real)
brng = np.random.default_rng(5); d = []
for _ in range(100):
    i = brng.integers(0, len(te), len(te)); d.append(roc_auc_score(yt[i], p1[i]) - roc_auc_score(yt[i], p0[i]))
lo, hi = np.percentile(d, [2.5, 97.5])
t0 = time.time(); g_ar = []; g_iid = []
for k in range(K):
    g_ar.append(gain(ar1(np.random.default_rng(100 + k)))[0]); g_iid.append(gain(np.random.default_rng(500 + k).permutation(real))[0])
    if k % 10 == 9: print(k + 1, f"{time.time()-t0:.0f}s", flush=True)
p_ar = (1 + sum(g >= g_real for g in g_ar)) / (K + 1); p_iid = (1 + sum(g >= g_real for g in g_iid)) / (K + 1)
res = dict(dataset=which, control=ctl, loans=len(m), statements=n, phi=phi, base_auc=a0, real_gain=g_real, boot_lo=lo, boot_hi=hi,
           standard_pass=int(lo > 0), p_ar=p_ar, p_iid=p_iid, mean_gain_ar=np.mean(g_ar), sd_gain_ar=np.std(g_ar), mean_gain_iid=np.mean(g_iid), sd_gain_iid=np.std(g_iid), K=K)
pd.DataFrame([res]).to_csv(OUT, index=False); print(res)
