"""
距离门控诊断: 空间重建质量 vs 最近邻井距离。
假说: 近邻井(<300-600ft)上spatial打败selector, 远井上spatial烂 -> 距离门控可救v6。
全部本地, 零提交成本。产出 distgate.pkl 供后续门控调参复用。
"""
import numpy as np, pickle, time, os
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
import heur, spatial_recon as sr

OUT = os.path.join(os.path.dirname(__file__), "distgate.pkl")

ids = heur.well_ids('train')
allX, allS, allW = sr.load_all_surface_samples(ids, stride=15)
tree = cKDTree(allX)
print(f"地层面样本 {len(allX)} ({len(set(allW))}井)", flush=True)

sc = {d['wid']: d for d in pickle.load(open('selcache.pkl', 'rb'))}

def one(wid):
    """返回 (wid, nn_dist, spatial_err数组, selector_err数组) 或 None"""
    try:
        r = sr.spatial_predict_well(wid, allX, allS, allW, tree, k=12, use_surf='EGFDU')
    except Exception:
        return None
    if r is None or wid not in sc:
        return None
    sp_err = r[0]
    d_sel = sc[wid]
    if len(sp_err) != len(d_sel['pred']):
        return None
    sel_err = d_sel['pred'] - d_sel['truth']
    # 最近邻井距离: 本井采样点到"其他井"采样点的最小距离(中位数更稳: 用本井各点nn距离的中位)
    mask = allW == wid
    pts = allX[mask][::4]  # 再抽稀加速
    if len(pts) == 0:
        return None
    dd, ii = tree.query(pts, k=200)
    nn_per_pt = []
    for j in range(len(pts)):
        other = allW[ii[j]] != wid
        if other.any():
            nn_per_pt.append(dd[j][other][0])
    if not nn_per_pt:
        return None
    nn_med = float(np.median(nn_per_pt))
    nn_min = float(np.min(nn_per_pt))
    return dict(wid=wid, nn_med=nn_med, nn_min=nn_min,
                sp_err=sp_err.astype(np.float32), sel_err=sel_err.astype(np.float32))

t0 = time.time()
res = [r for r in Parallel(n_jobs=-1)(delayed(one)(w) for w in sc.keys()) if r]
pickle.dump(res, open(OUT, 'wb'))
print(f"完成 {len(res)}井 {time.time()-t0:.0f}s -> {OUT}", flush=True)

# ---- 诊断: 按 nn_min 分桶 ----
nn = np.array([r['nn_min'] for r in res])
sp_rmse = np.array([np.sqrt(np.mean(r['sp_err']**2)) for r in res])
sel_rmse = np.array([np.sqrt(np.mean(r['sel_err']**2)) for r in res])
print(f"\n最近邻距离分布: p10={np.percentile(nn,10):.0f} p25={np.percentile(nn,25):.0f} "
      f"p50={np.percentile(nn,50):.0f} p75={np.percentile(nn,75):.0f} p90={np.percentile(nn,90):.0f}")

print(f"\n{'距离桶':>14s} {'井数':>4s} {'spatial':>8s} {'selector':>8s} {'sp更好井%':>8s} {'0.85sel+0.15sp':>14s}")
bins = [(0,150),(150,300),(300,600),(600,1200),(1200,3000),(3000,1e9)]
for lo,hi in bins:
    m = (nn>=lo)&(nn<hi)
    if m.sum()==0: continue
    sub = [r for r,mm in zip(res,m) if mm]
    e_sp = np.concatenate([r['sp_err'] for r in sub]); e_sel = np.concatenate([r['sel_err'] for r in sub])
    e_mix = np.concatenate([0.85*r['sel_err']+0.15*r['sp_err'] for r in sub])
    better = np.mean([np.sqrt(np.mean(r['sp_err']**2))<np.sqrt(np.mean(r['sel_err']**2)) for r in sub])
    print(f"{lo:>6.0f}-{hi:<7.0f} {m.sum():>4d} {np.sqrt(np.mean(e_sp**2)):>8.3f} {np.sqrt(np.mean(e_sel**2)):>8.3f} "
          f"{better*100:>7.0f}% {np.sqrt(np.mean(e_mix**2)):>14.3f}")

print(f"\ncorr(nn_min, sp_rmse - sel_rmse) = {np.corrcoef(nn, sp_rmse-sel_rmse)[0,1]:+.3f}  (正=越远spatial相对越差, 支持门控)")
