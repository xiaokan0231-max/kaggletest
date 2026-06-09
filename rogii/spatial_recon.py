"""
方向②探路: 留一井空间重建。
地层面(EGFDU等)空间连续->从邻井IDW插值; TVT = C-(Z-面_interp), C用已知TVT_input段标定。
无泄漏(不用本井地层面/TVT)。对比 selector(~10) 和 本井面物理模型(泄漏~0)。
"""
import pandas as pd, numpy as np, heur, glob, os, time
from scipy.spatial import cKDTree

SURF = ['ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA']
DATA = heur.DATA


def load_all_surface_samples(ids, stride=15):
    """收集所有训练井的(X,Y,6面)采样点 + well标签。"""
    X = []; S = []; W = []
    for wid in ids:
        try:
            h = pd.read_csv(f'{DATA}/train/{wid}__horizontal_well.csv')
        except Exception:
            continue
        if not all(c in h.columns for c in SURF):
            continue
        sub = h.iloc[::stride]
        X.append(sub[['X', 'Y']].values)
        S.append(sub[SURF].values)
        W.append(np.full(len(sub), wid, dtype=object))
    return np.vstack(X), np.vstack(S), np.concatenate(W)


def spatial_predict_well(wid, allX, allS, allW, tree, k=12, use_surf='EGFDU', multi=False):
    h = pd.read_csv(f'{DATA}/train/{wid}__horizontal_well.csv')
    ps = int(h['TVT_input'].notna().sum())
    if ps < 60 or len(h) - ps < 30:
        return None
    truth = h['TVT'].values[ps:].astype(float)
    if np.isnan(truth).any():
        return None
    Z = h['Z'].values.astype(float)
    pts = h[['X', 'Y']].values
    tvti = h['TVT_input'].values.astype(float)
    # 对每个点, 邻近k个非本井样本 IDW 插值地层面
    # 取大量近邻, 剔本井后保证够k个(密集区本井点会占满近邻)
    nq = min(len(allW), k + 400)
    d, idx = tree.query(pts, k=nq)
    si = SURF.index(use_surf)
    surf_interp = np.full((len(h), len(SURF)), np.nan)
    for i in range(len(h)):
        ii = idx[i]; dd = d[i]
        keep = allW[ii] != wid       # 剔除本井
        ii = ii[keep][:k]; dd = dd[keep][:k]
        if len(ii) == 0:
            return None
        w = 1.0 / (dd + 1.0)
        w /= w.sum()
        surf_interp[i] = (w[:, None] * allS[ii]).sum(0)
    # 物理模型: TVT = C - (Z - surf)
    if multi:
        # 各面给一个TVT估计, 各自标定C, 再平均
        preds = []
        for j in range(len(SURF)):
            raw = -(Z - surf_interp[:, j])
            C = np.nanmean(tvti[:ps] - raw[:ps])
            preds.append(raw + C)
        pred = np.nanmean(np.stack(preds, 0), 0)
    else:
        raw = -(Z - surf_interp[:, si])
        C = np.nanmean(tvti[:ps] - raw[:ps])
        pred = raw + C
    return pred[ps:] - truth, np.full_like(truth, float(tvti[ps - 1])) - truth


def cv(n=60, seed=0, k=12, use_surf='EGFDU', multi=False, stride=15):
    import random
    ids = heur.well_ids('train')
    random.seed(seed); samp = random.sample(ids, min(n, len(ids)))
    t0 = time.time()
    print(f"收集地层面样本(stride={stride})...", flush=True)
    allX, allS, allW = load_all_surface_samples(ids, stride=stride)
    tree = cKDTree(allX)
    print(f"  {len(allX)} 样本点, {time.time()-t0:.0f}s. 开始留一井评估...", flush=True)
    e_sp, e_c = [], []
    for wid in samp:
        r = spatial_predict_well(wid, allX, allS, allW, tree, k=k, use_surf=use_surf, multi=multi)
        if r is None:
            continue
        e_sp.append(r[0]); e_c.append(r[1])
    e_sp = np.concatenate(e_sp); e_c = np.concatenate(e_c)
    return dict(spatial=float(np.sqrt(np.mean(e_sp**2))),
                const=float(np.sqrt(np.mean(e_c**2))),
                wells=len(e_c) and len(e_sp), secs=time.time()-t0)


if __name__ == "__main__":
    print("=== 留一井空间重建探路 ===", flush=True)
    for surf, multi in [('EGFDU', False), ('ANCC', False), (None, True)]:
        r = cv(n=80, seed=0, k=12, use_surf=surf or 'EGFDU', multi=multi)
        tag = 'multi-surf平均' if multi else f'单面{surf}'
        print(f"{tag:16s}: spatial RMSE={r['spatial']:.4f}  const={r['const']:.4f}  {r['secs']:.0f}s", flush=True)
    print("\n对比基准: selector(无泄漏)≈10.5, 本井面物理模型(泄漏)≈0.005, fork公榜7.904")
