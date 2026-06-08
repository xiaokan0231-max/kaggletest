"""
粒子滤波地质导向跟踪器 (纯 CPU)。
状态: pos = TVT + Z (构造高程), rate = d(pos)/dMD。
- 运动: rate <- mom*rate + noise;  pos <- pos + rate*dmd + noise   (mom<1 -> 漂移阻尼饱和)
- 观测: TVT = pos - Z(已知) -> 查参考井 GR(TVT) 与水平井标定GR比 -> 似然加权重采样
- 估计: TVT_hat = E[pos] - Z;  最后 Savgol 平滑
"""
import numpy as np, pandas as pd, glob, os
from scipy.signal import savgol_filter

DATA = os.path.join(os.path.dirname(__file__), "data", "extracted")


def _calibrate(h, t, ps):
    ts = t.dropna(subset=["TVT", "GR"]).sort_values("TVT")
    tvt_ref, gr_ref = ts["TVT"].values.astype(float), ts["GR"].values.astype(float)
    kn = h.iloc[:ps].dropna(subset=["GR", "TVT"])
    if len(kn) >= 20:
        gtk = np.interp(kn["TVT"].values, tvt_ref, gr_ref)
        a, b = np.polyfit(kn["GR"].values, gtk, 1)
        resid = (a * kn["GR"].values + b) - gtk
        sig = float(np.clip(np.std(resid), 8.0, 60.0))
    else:
        a, b, sig = 1.0, 0.0, 30.0
    return a, b, tvt_ref, gr_ref, sig


def run_pf(h, t, N=600, mom=0.997, q_pos=0.03, q_rate=0.004,
           init_pos=4.0, init_rate=0.02, rough_pos=0.05, rough_rate=0.002,
           sg_win=31, sg_poly=3, seed=42, sig_scale=1.0, ir_w=1.0,
           return_parts=False):
    ti = h["TVT_input"].values.astype(float)
    ps = int(np.sum(~np.isnan(ti)))
    out = ti.copy()
    n = len(h)
    if ps == 0:
        out[:] = np.nanmean(t["TVT"]); return out
    if ps >= n:
        return out
    a, b, tvt_ref, gr_ref, sig = _calibrate(h, t, ps)
    md = h["MD"].values.astype(float); Z = h["Z"].values.astype(float)
    gh = a * h["GR"].values.astype(float) + b  # 标定到参考井GR空间

    # 初始速率 ir = d(TVT+Z)/dMD 末30点
    k = min(30, ps - 1)
    ls_known = ti[:ps] + Z[:ps]
    if k >= 3:
        dls = np.diff(ls_known[-k - 1:]); dm = np.diff(md[ps - k - 1:ps])
        m = dm > 0
        ir = float(np.median(dls[m] / dm[m])) if m.sum() >= 3 else 0.0
    else:
        ir = 0.0
    ir *= ir_w
    ls0 = ti[ps - 1] + Z[ps - 1]

    rng = np.random.default_rng(seed)
    pos = ls0 + init_pos * rng.standard_normal(N)
    rate = ir + init_rate * rng.standard_normal(N)
    w = np.full(N, 1.0 / N)

    sig *= sig_scale
    est = np.empty(n - ps)
    inv2s2 = 1.0 / (2.0 * sig * sig)
    for j in range(ps, n):
        dmd = md[j] - md[j - 1]
        # 运动
        rate = mom * rate + q_rate * rng.standard_normal(N)
        pos = pos + rate * dmd + q_pos * rng.standard_normal(N)
        # 观测更新
        g = gh[j]
        if not np.isnan(g):
            tvt_p = pos - Z[j]
            gr_pred = np.interp(tvt_p, tvt_ref, gr_ref)
            logw = -((g - gr_pred) ** 2) * inv2s2
            logw -= logw.max()
            w = w * np.exp(logw)
            s = w.sum()
            if s <= 0 or not np.isfinite(s):
                w = np.full(N, 1.0 / N)
            else:
                w /= s
            # 重采样(ESS 低时)
            ess = 1.0 / np.sum(w * w)
            if ess < 0.5 * N:
                idx = _systematic(w, rng)
                pos = pos[idx] + rough_pos * rng.standard_normal(N)
                rate = rate[idx] + rough_rate * rng.standard_normal(N)
                w = np.full(N, 1.0 / N)
        est[j - ps] = np.sum(w * pos) - Z[j]

    # Savgol 平滑
    if sg_win and len(est) >= sg_win:
        wl = sg_win if sg_win % 2 == 1 else sg_win - 1
        if wl >= sg_poly + 2:
            est = savgol_filter(est, wl, sg_poly)
    out[ps:] = est
    return out


def _systematic(w, rng):
    N = len(w)
    pos = (rng.random() + np.arange(N)) / N
    cum = np.cumsum(w)
    return np.searchsorted(cum, pos)


# ---------- CV ----------
def _load(folder, wid):
    h = pd.read_csv(os.path.join(folder, f"{wid}__horizontal_well.csv"))
    t = pd.read_csv(glob.glob(os.path.join(folder, f"{wid}__typewell*.csv"))[0])
    return h, t
def _ids(folder):
    return sorted(os.path.basename(f).split("__")[0]
                  for f in glob.glob(os.path.join(folder, "*__horizontal_well.csv")))


def cv(n=60, seed=0, **kw):
    import random, time
    ids = _ids(os.path.join(DATA, "train"))
    random.seed(seed); samp = random.sample(ids, n)
    e_pf, e_c, imp = [], [], 0
    t0 = time.time()
    for wid in samp:
        h, t = _load(os.path.join(DATA, "train"), wid)
        ps = h["TVT_input"].notna().sum()
        if ps < 60 or len(h) - ps < 30:
            continue
        truth = h["TVT"].values[ps:]
        pred = run_pf(h, t, **kw)[ps:]
        const = h["TVT_input"].values[ps - 1]
        rp = np.sqrt(np.mean((pred - truth) ** 2))
        rc = np.sqrt(np.mean((const - truth) ** 2))
        e_pf.append(pred - truth); e_c.append(const - truth); imp += (rp < rc)
    e_pf = np.concatenate(e_pf); e_c = np.concatenate(e_c)
    return dict(pf=np.sqrt(np.mean(e_pf**2)), const=np.sqrt(np.mean(e_c**2)),
                nwell=len(e_c) and imp, t=time.time()-t0)


if __name__ == "__main__":
    r = cv(n=60, seed=0)
    print(f"PF baseline: PF={r['pf']:.3f}  const={r['const']:.3f}  改善井数={r['nwell']}  {r['t']:.1f}s")
