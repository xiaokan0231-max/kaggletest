"""
把全量训练井的 selector 原始预测算一次缓存下来,
之后任意后处理(hold/平滑/锚点)在全 773 井上秒级评估 -> 消除样本依赖, 给铁定结论。
"""
import heur, numpy as np, pickle, os, time
from joblib import Parallel, delayed
from scipy.signal import savgol_filter

CACHE = os.path.join(os.path.dirname(__file__), "selcache.pkl")


def _raw_one(wid, n_seeds, min_ps, min_ev):
    try:
        hw, tw = heur.load_well(wid, 'train')
    except Exception:
        return None
    if 'TVT' not in hw.columns:
        return None
    ps = int(hw['TVT_input'].notna().sum())
    if ps < min_ps or len(hw) - ps < min_ev:
        return None
    truth = hw['TVT'].values[ps:].astype(float)
    if np.isnan(truth).any():
        return None
    pred = heur.selector_predict(hw, tw, n_seeds=n_seeds)[ps:].astype(float)
    kn = hw['TVT_input'].dropna()
    last = float(kn.iloc[-1]) if len(kn) else float(np.nanmean(pred))
    md = hw['MD'].values.astype(float)
    md_since = np.maximum(md[ps:] - md[ps - 1], 0.0)
    return dict(wid=wid, pred=pred, truth=truth, last=last, md_since=md_since)


def build_cache(n=None, n_seeds=32, n_jobs=-1, min_ps=60, min_ev=30):
    ids = heur.well_ids('train')
    if n:
        ids = ids[:n]
    t0 = time.time()
    res = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_raw_one)(w, n_seeds, min_ps, min_ev) for w in ids)
    res = [r for r in res if r is not None]
    pickle.dump(res, open(CACHE, 'wb'))
    print(f"cached {len(res)} wells in {time.time()-t0:.1f}s -> {CACHE}", flush=True)
    return res


def load_cache():
    return pickle.load(open(CACHE, 'rb'))


def pooled_rmse(cache, postproc=None):
    """postproc(d)->处理后的预测段; None=原始。返回全井 pooled RMSE。"""
    errs = []
    for d in cache:
        p = d['pred'] if postproc is None else postproc(d)
        errs.append(p - d['truth'])
    e = np.concatenate(errs)
    return float(np.sqrt(np.mean(e**2)))


# ---- 后处理算子 (作用于缓存的原始预测) ----
def pp_hold(h):
    def f(d): return (1 - h) * d['pred'] + h * d['last']
    return f

def pp_revhold(hmax, tau):
    def f(d):
        w = hmax * (1 - np.exp(-d['md_since'] / tau))
        return (1 - w) * d['pred'] + w * d['last']
    return f

def pp_smooth(sg_w, sg_p=3):
    def f(d):
        seg = d['pred']
        if len(seg) >= sg_p + 2:
            wl = min(sg_w, len(seg)); wl = wl if wl % 2 == 1 else wl - 1
            if wl >= sg_p + 2:
                seg = savgol_filter(seg, wl, sg_p)
        return seg
    return f

def pp_smooth_hold(sg_w, h, sg_p=3):
    sm = pp_smooth(sg_w, sg_p)
    def f(d):
        seg = sm(d)
        return (1 - h) * seg + h * d['last']
    return f


if __name__ == "__main__":
    import sys
    if not os.path.exists(CACHE) or (len(sys.argv) > 1 and sys.argv[1] == 'rebuild'):
        build_cache(n_seeds=32)
    cache = load_cache()
    n_pts = sum(len(d['truth']) for d in cache)
    base = pooled_rmse(cache)
    print(f"全量缓存: {len(cache)} 井, {n_pts} 点")
    print(f"baseline pooled RMSE = {base:.4f}")
    print("--- hold 扫描 (全井) ---")
    for h in [0.0, 0.02, 0.03, 0.05, 0.07, 0.10, 0.13, 0.16, 0.20]:
        r = pooled_rmse(cache, pp_hold(h))
        print(f"  hold {h:.2f} = {r:.4f}  (Δ {r-base:+.4f})")
    print("--- revhold 扫描 ---")
    for hmax in [0.10, 0.15, 0.20, 0.25]:
        for tau in [400., 800.]:
            r = pooled_rmse(cache, pp_revhold(hmax, tau))
            print(f"  revhold hmax={hmax} tau={tau:.0f} = {r:.4f}  (Δ {r-base:+.4f})")
    print("--- smooth+hold ---")
    for sg in [31, 51]:
        for h in [0.03, 0.05, 0.08]:
            r = pooled_rmse(cache, pp_smooth_hold(sg, h))
            print(f"  sg{sg}+hold{h:.2f} = {r:.4f}  (Δ {r-base:+.4f})")
