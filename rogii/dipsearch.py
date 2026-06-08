"""
Dip-search aligner: 预测区 TVT = TVT_ps + dip*(MD-MD_ps)。
对每口井搜索使"水平井 GR(经已知段标定) 匹配参考井 GR(TVT)"代价最小的 dip。
鲁棒: 1维搜索, 平缓井 -> dip≈0(=常数), 倾斜井 -> 抓住斜率。
"""
import numpy as np, pandas as pd, glob, os


def calibrate(h, t, ps):
    """用已知段拟合 gr_type(TVT) ≈ a*gr_horiz + b, 返回 (a,b, tvt_ref, gr_ref, corr)。"""
    ts = t.dropna(subset=["TVT", "GR"]).sort_values("TVT")
    tvt_ref, gr_ref = ts["TVT"].values, ts["GR"].values
    kn = h.iloc[:ps].dropna(subset=["GR", "TVT"])
    if len(kn) < 20:
        return 1.0, 0.0, tvt_ref, gr_ref, 0.0
    gtype_known = np.interp(kn["TVT"].values, tvt_ref, gr_ref)
    a, b = np.polyfit(kn["GR"].values, gtype_known, 1)
    corr = np.corrcoef(a * kn["GR"].values + b, gtype_known)[0, 1]
    return a, b, tvt_ref, gr_ref, corr


def predict_dip(h, t, dmax=0.05, ndip=121, return_dip=False):
    ti = h["TVT_input"].values.astype(float)
    ps = int(np.sum(~np.isnan(ti)))
    out = ti.copy()
    n = len(h)
    if ps == 0:
        out[:] = np.nanmean(t["TVT"]); return (out, 0.0) if return_dip else out
    if ps >= n:
        return (out, 0.0) if return_dip else out
    a, b, tvt_ref, gr_ref, corr = calibrate(h, t, ps)
    md = h["MD"].values; tps = ti[ps - 1]; mps = md[ps - 1]
    dmd = md[ps:] - mps
    gh = a * h["GR"].values[ps:] + b
    valid = ~np.isnan(gh)
    dips = np.linspace(-dmax, dmax, ndip)
    best_d, best_c = 0.0, np.inf
    ghv = gh[valid]; dmdv = dmd[valid]
    for d in dips:
        tvt_pred = tps + d * dmdv
        gt = np.interp(tvt_pred, tvt_ref, gr_ref)
        c = np.mean((ghv - gt) ** 2)
        if c < best_c:
            best_c, best_d = c, d
    out[ps:] = tps + best_d * dmd
    return (out, best_d) if return_dip else out


# ---------- 评估 ----------
DATA = os.path.join(os.path.dirname(__file__), "data", "extracted")
def _load(folder, wid):
    h = pd.read_csv(os.path.join(folder, f"{wid}__horizontal_well.csv"))
    t = pd.read_csv(glob.glob(os.path.join(folder, f"{wid}__typewell*.csv"))[0])
    return h, t
def _ids(folder):
    return sorted(os.path.basename(f).split("__")[0]
                  for f in glob.glob(os.path.join(folder, "*__horizontal_well.csv")))


def cv(n=120, seed=0, **kw):
    import random
    ids = _ids(os.path.join(DATA, "train"))
    random.seed(seed); samp = random.sample(ids, n)
    e_dp, e_c, improved, true_dip, pred_dip = [], [], 0, [], []
    for wid in samp:
        h, t = _load(os.path.join(DATA, "train"), wid)
        ps = h["TVT_input"].notna().sum()
        if ps < 60 or len(h) - ps < 30:
            continue
        truth = h["TVT"].values[ps:]
        pred, d = predict_dip(h, t, return_dip=True, **kw)
        pred = pred[ps:]
        const = h["TVT_input"].values[ps - 1]
        rd = np.sqrt(np.mean((pred - truth) ** 2))
        rc = np.sqrt(np.mean((const - truth) ** 2))
        e_dp.append(pred - truth); e_c.append(const - truth)
        improved += (rd < rc)
        # 真实 dip = 真值线性拟合斜率
        td = np.polyfit(h["MD"].values[ps:] - h["MD"].values[ps-1], truth, 1)[0]
        true_dip.append(td); pred_dip.append(d)
    e_dp = np.concatenate(e_dp); e_c = np.concatenate(e_c)
    nwell = len(true_dip)
    dipcorr = np.corrcoef(true_dip, pred_dip)[0, 1]
    return dict(dp=np.sqrt(np.mean(e_dp**2)), const=np.sqrt(np.mean(e_c**2)),
                nwell=nwell, improved=improved, dipcorr=dipcorr)


if __name__ == "__main__":
    import time
    for dmax in [0.03, 0.05, 0.08]:
        t0 = time.time()
        r = cv(n=120, dmax=dmax, ndip=161)
        print(f"dmax={dmax}: DP={r['dp']:.3f} const={r['const']:.3f} "
              f"| 改善井 {r['improved']}/{r['nwell']} | dip相关={r['dipcorr']:.3f} | {time.time()-t0:.1f}s")
