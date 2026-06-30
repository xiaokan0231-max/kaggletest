"""
方向②探路: 留一井空间重建 (WLS 局部平面拟合版本)。
地层面(EGFDU等)空间连续 -> 从邻井拟合局部倾斜平面 -> TVT = C - (Z - 面_pred)，C用已知TVT_input段标定。
无泄漏(不用本井地层面/TVT)。
"""
import pandas as pd
import numpy as np
import heur
import glob
import os
import time
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression

SURF = ['ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA']
DATA = heur.DATA

def load_all_surface_samples(ids, stride=15):
    """收集所有训练井的(X,Y,6面)采样点 + well标签。"""
    X = []
    S = []
    W = []
    for wid in ids:
        try:
            h = pd.read_csv(f'{DATA}/train/{wid}__horizontal_well.csv')
        except Exception:
            continue
        if not all(c in h.columns for c in SURF):
            continue
        sub = h.dropna(subset=SURF).iloc[::stride]
        if len(sub) == 0:
            continue
        X.append(sub[['X', 'Y']].values)
        S.append(sub[SURF].values)
        W.append(np.full(len(sub), wid, dtype=object))
    return np.vstack(X), np.vstack(S), np.concatenate(W)

def cv(n=80, seed=0, k_wells=5, p=2.0, multi=True):
    import random
    ids = heur.well_ids('train')
    
    # 收集样本并构建井级空间结构
    allX, allS, allW = load_all_surface_samples(ids, stride=15)
    
    well_centers = {}
    well_points_dict = {}
    for wid in ids:
        try:
            h = pd.read_csv(f'{DATA}/train/{wid}__horizontal_well.csv')
        except Exception:
            continue
        sub = h.dropna(subset=SURF)
        if len(sub) > 0:
            well_centers[wid] = sub[['X', 'Y']].mean().values
            well_points_dict[wid] = {s: sub[['X', 'Y', s]].values for s in SURF}
            
    well_names = list(well_centers.keys())
    well_coords = np.array([well_centers[w] for w in well_names])
    well_tree = cKDTree(well_coords)
    
    # 抽样评估井
    random.seed(seed)
    samp = random.sample(well_names, min(n, len(well_names)))
    
    e_sp, e_c = [], []
    t0 = time.time()
    
    for wid in samp:
        h = pd.read_csv(f'{DATA}/train/{wid}__horizontal_well.csv')
        ps = int(h['TVT_input'].notna().sum())
        if ps < 60 or len(h) - ps < 30:
            continue
        truth = h['TVT'].values[ps:].astype(float)
        if np.isnan(truth).any():
            continue
        Z = h['Z'].values.astype(float)
        pts = h[['X', 'Y']].values
        tvti = h['TVT_input'].values.astype(float)
        
        # 查询最近的邻近井
        q_center = well_centers[wid]
        wd, w_idx = well_tree.query(q_center, k=k_wells + 1)
        wd = np.atleast_1d(wd)
        w_idx = np.atleast_1d(w_idx)
        
        # 排除当前评估井
        keep = [idx for idx in w_idx if well_names[idx] != wid][:k_wells]
        keep_wids = [well_names[idx] for idx in keep]
        keep_dists = [wd[list(w_idx).index(idx)] for idx in keep]
        
        surfs_to_use = SURF if multi else ['EGFDU']
        preds = []
        
        for s in surfs_to_use:
            nb_pts = []
            weights_list = []
            for n_wid, dist in zip(keep_wids, keep_dists):
                pts_s = well_points_dict[n_wid][s]
                nb_pts.append(pts_s)
                w_val = 1.0 / (dist + 1.0)**p if p > 0 else 1.0
                weights_list.append(np.full(len(pts_s), w_val))
            
            nb_pts = np.vstack(nb_pts)
            weights = np.concatenate(weights_list)
            
            lr = LinearRegression()
            lr.fit(nb_pts[:, :2], nb_pts[:, 2], sample_weight=weights)
            pred_surf = lr.predict(pts)
            
            raw = -(Z - pred_surf)
            C = np.nanmean(tvti[:ps] - raw[:ps])
            preds.append(raw + C)
            
        pred_final = np.nanmean(np.stack(preds, 0), 0)
        
        e_sp.append(pred_final[ps:] - truth)
        e_c.append(np.full_like(truth, tvti[ps - 1]) - truth)
        
    e_sp = np.concatenate(e_sp)
    e_c = np.concatenate(e_c)
    
    return dict(spatial=float(np.sqrt(np.mean(e_sp**2))),
                const=float(np.sqrt(np.mean(e_c**2))),
                wells=len(samp), secs=time.time()-t0)

if __name__ == "__main__":
    print("=== 留一井空间重建 (WLS 局部平面拟合版本) ===", flush=True)
    r = cv(n=100, seed=0, k_wells=5, p=2.0, multi=True)
    print(f"WLS Local Plane Fit (k=5, p=2.0, multi=True): spatial RMSE={r['spatial']:.4f}  const={r['const']:.4f}  ({r['secs']:.1f}s)", flush=True)
