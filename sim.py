"""Paper A simulation: how often is the standard 'AUC gain' check fooled by a meaningless time series,
and does a placebo-based (empirical-null) test fix it?  Plain logistic scorecard on simulated loans."""
import numpy as np, sys, time
from scipy.stats import rankdata

def auc(y, p):
    r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def fit_logit(X, y, b0=None, it=8):
    """Newton / IRLS logistic regression with a tiny ridge; warm start allowed."""
    b = np.zeros(X.shape[1]) if b0 is None else b0.copy()
    for _ in range(it):
        z = X @ b; p = 1 / (1 + np.exp(-z)); w = p * (1 - p) + 1e-9
        g = X.T @ (y - p) - 1e-4 * b
        H = (X * w[:, None]).T @ X + 1e-4 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g); b += step
        if np.abs(step).max() < 1e-6: break
    return b

def ar1(T, phi, rng):
    e = rng.standard_normal(T); z = np.zeros(T); z[0] = e[0]
    for t in range(1, T): z[t] = phi * z[t - 1] + e[t] * np.sqrt(1 - phi ** 2)
    return (z - z.mean()) / z.std()

def simulate(gamma, phi_s, rng, T=40, per_q=600, phi_c=0.9, sd_c=0.5):
    n = T * per_q
    q = np.repeat(np.arange(T), per_q); yr = q // 4
    X = rng.standard_normal((n, 5)); beta = np.array([.8, -.6, .5, .4, -.3])
    c = sd_c * ar1(T, phi_c, rng)                 # unobserved credit cycle
    s = ar1(T, phi_s, rng)                         # the candidate time series (independent of c)
    lin = -3.0 + X @ beta + c[q] + gamma * s[q]
    y = (rng.random(n) < 1 / (1 + np.exp(-lin))).astype(float)
    return X, y, q, yr, s

def one_rep(gamma, phi_s, seed, K=99, B=100, T=40, per_q=600):
    rng = np.random.default_rng(seed)
    X, y, q, yr, s = simulate(gamma, phi_s, rng, T, per_q)
    n = len(y)
    D = (yr[:, None] == np.arange(yr.max() + 1)[None, :]).astype(float)       # year dummies
    Xb = np.hstack([X, D])
    idx = rng.permutation(n); ntr = int(.7 * n); tr, te = idx[:ntr], idx[ntr:]
    b0 = fit_logit(Xb[tr], y[tr]); p0 = 1 / (1 + np.exp(-(Xb[te] @ b0))); yt = y[te]
    a0 = auc(yt, p0)
    def gain_of(series):
        v = series[q]; v = v - np.array([v[yr == k].mean() for k in range(yr.max() + 1)])[yr]   # within-year part
        Z = np.hstack([Xb, v[:, None]])
        b = fit_logit(Z[tr], y[tr], np.append(b0, 0.0))
        p = 1 / (1 + np.exp(-(Z[te] @ b)))
        return auc(yt, p) - a0, p
    g_real, p1 = gain_of(s)
    # (1) standard check: bootstrap the TEST SET only
    brng = np.random.default_rng(seed + 7); d = []
    for _ in range(B):
        i = brng.integers(0, len(te), len(te)); d.append(auc(yt[i], p1[i]) - auc(yt[i], p0[i]))
    std_pass = np.percentile(d, 2.5) > 0
    # (2) empirical-null with iid placebos ; (3) with persistence-matched placebos
    phi_hat = np.corrcoef(s[:-1], s[1:])[0, 1]
    g_iid = [gain_of(rng.permutation(s))[0] for _ in range(K)]
    g_ar = [gain_of(ar1(T, max(phi_hat, 0), rng))[0] for _ in range(K)]
    p_iid = (1 + sum(g >= g_real for g in g_iid)) / (K + 1)
    p_ar = (1 + sum(g >= g_real for g in g_ar)) / (K + 1)
    return dict(gamma=gamma, phi_s=phi_s, seed=seed, gain=g_real, std_pass=int(std_pass),
                p_iid=p_iid, p_ar=p_ar, phi_hat=phi_hat)

if __name__ == "__main__":
    t = time.time(); r = one_rep(0.0, 0.9, 1); print(r, f"{time.time()-t:.1f}s")
