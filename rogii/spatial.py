"""
空间构造面重建 (collaborative structural mapping):
TVT_station ≈ S(X,Y) + c_well   (S=共享构造面/dip场, c=每井地层基准)
- 训练井: 去基准得结构偏差 v = TVT - c_well, 建 (X,Y)->v 的 KNN 场
- 测试井: 用已知段把结构面对齐 (求 c_test), 预测区 TVT = S(X,Y) + c_test
打败常数基线的关键: 邻井先去基准再聚合, 否则不同地层偏移糊掉 dip 信号。
"""
import numpy as np, pandas as pd, glob, os, time
from scipy.spatial import cKDTree

DATA = os.path.join(os.path.dirname(__file__), "data", "extracted")
TR = os.path.join(DATA, "train")


def load_all(folder, step=3):
    ids = sorted(os.path.basename(f).split("__")[0]
                 for f in glob.glob(os.path.join(folder, "*__horizontal_well.csv")))
    wells = {}
    for wid in ids:
        h = pd.read_csv(os.path.join(folder, f"{wid}__horizontal_well.csv"),
                        usecols=["MD", "X", "Y", "Z", "GR", "TVT", "TVT_input"])
        wells[wid] = h
    return ids, wells


def build_field(ids, wells, step=3, exclude=None):
    """返回 KDTree + 去基准结构值 v + 来源井索引。c_well = 全井 TVT 均值。"""
    Xs, Ys, Vs, Ws = [], [], [], []
    for k, wid in enumerate(ids):
        if wid == exclude:
            continue
        h = wells[wid]
        tvt = h["TVT"].values
        c = np.nanmean(tvt)
        s = slice(None, None, step)
        Xs.append(h["X"].values[s]); Ys.append(h["Y"].values[s])
        Vs.append(tvt[s] - c); Ws.append(np.full(len(tvt[s]), k))
    X = np.concatenate(Xs); Y = np.concatenate(Ys)
    V = np.concatenate(Vs); W = np.concatenate(Ws)
    tree = cKDTree(np.c_[X, Y])
    return tree, V, W


def knn_struct(tree, V, qx, qy, k=16):
    d, ii = tree.query(np.c_[qx, qy], k=k)
    d = np.atleast_2d(d); ii = np.atleast_2d(ii)
    w = 1.0 / (d + 1.0)
    vals = V[ii]
    return np.sum(w * vals, axis=1) / np.sum(w, axis=1)


def predict_spatial(h, tree, V, k=16):
    ti = h["TVT_input"].values.astype(float)
    ps = int(np.sum(~np.isnan(ti)))
    out = ti.copy()
    if ps == 0 or ps >= len(h):
        out[np.isnan(out)] = np.nanmean(ti); return out
    x = h["X"].values; y = h["Y"].values
    # 已知段对齐 -> c_test
    s_known = knn_struct(tree, V, x[:ps], y[:ps], k=k)
    c_test = np.nanmean(ti[:ps] - s_known)
    # 预测区
    s_pred = knn_struct(tree, V, x[ps:], y[ps:], k=k)
    out[ps:] = s_pred + c_test
    return out


def cv(n=60, seed=7, step=3, k=16, blend_const=0.0):
    import random
    ids, wells = load_all(TR, step)
    random.seed(seed); samp = random.sample(ids, n)
    e_sp, e_c = [], []
    for wid in samp:
        h = wells[wid]; ps = h["TVT_input"].notna().sum()
        if ps < 60 or len(h) - ps < 30:
            continue
        tree, V, W = build_field(ids, wells, step=step, exclude=wid)
        pred = predict_spatial(h, tree, V, k=k)[ps:]
        const = h["TVT_input"].values[ps - 1]
        if blend_const > 0:
            pred = (1 - blend_const) * pred + blend_const * const
        truth = h["TVT"].values[ps:]
        e_sp.append(pred - truth); e_c.append(const - truth)
    e_sp = np.concatenate(e_sp); e_c = np.concatenate(e_c)
    return np.sqrt(np.mean(e_sp**2)), np.sqrt(np.mean(e_c**2))


if __name__ == "__main__":
    for k in [8, 16, 32]:
        t0 = time.time()
        sp, c = cv(n=60, k=k)
        print(f"k={k:2d}: 空间={sp:.3f}  常数={c:.3f}  ({time.time()-t0:.1f}s)")
