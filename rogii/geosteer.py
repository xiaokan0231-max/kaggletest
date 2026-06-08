"""
Geosteering DP aligner: 预测水平井 PS 之后的 TVT。
思路: 在已知 TVT(PS) 锚点之上, 寻找 TVT(MD) 路径, 使水平井归一化 GR 匹配
参考井 GR(TVT) 曲线, 同时一阶差分(dip)平滑。动态规划全局求解避开多解。

cost = Σ_i  data[i, tvt_i]              # GR 匹配 (仅 GR 非 NaN 的站)
     + Σ_i  lam * (tvt_i - tvt_{i-1})^2 # 平滑/连续
data[i,g] = (zGR_horiz[i] - zGR_type(g))^2
"""
import numpy as np, pandas as pd, glob, os


def _z(x):
    x = np.asarray(x, float)
    m = np.nanmean(x); s = np.nanstd(x)
    return (x - m) / (s + 1e-9)


def align_well(h, t, lam=1.0, res=1.0, half_range=120.0, kmax=3):
    """返回长度 len(h) 的 TVT 预测(已知段用真值, 预测段用 DP)。"""
    ti = h["TVT_input"].values.astype(float)
    known_mask = ~np.isnan(ti)
    ps = int(known_mask.sum())
    out = ti.copy()
    n = len(h)
    if ps >= n:
        return out
    if ps == 0:
        out[:] = np.nanmean(t["TVT"]); return out

    # 参考井 GR(TVT)
    ts = t.dropna(subset=["TVT", "GR"]).sort_values("TVT")
    tvt_ref, gr_ref = ts["TVT"].values, ts["GR"].values
    zref_fn_x = tvt_ref
    zref_fn_y = _z(gr_ref)

    # 水平井 GR 归一化(整井统计)
    zgr_h = _z(h["GR"].values)

    last_tvt = ti[ps - 1]
    # TVT 候选网格(围绕最后已知值)
    grid = np.arange(last_tvt - half_range, last_tvt + half_range + res, res)
    G = len(grid)
    # 每个网格点的参考 zGR
    zref_grid = np.interp(grid, zref_fn_x, zref_fn_y,
                          left=zref_fn_y[0], right=zref_fn_y[-1])

    # DP 仅在预测段 [ps:]
    idx_pred = np.arange(ps, n)
    Np = len(idx_pred)
    # 初始: 从已知锚点 last_tvt 的转移代价(parabola)
    prev = lam * (grid - last_tvt) ** 2
    # 加第 ps 站的发射
    g0 = zgr_h[ps]
    if not np.isnan(g0):
        prev = prev + (g0 - zref_grid) ** 2
    back = np.empty((Np, G), dtype=np.int32)
    back[0] = np.arange(G)
    # 转移核: 窗口 ±kmax, 惩罚 lam*(k*res)^2
    ks = np.arange(-kmax, kmax + 1)
    pen = lam * (ks * res) ** 2  # 长度 2kmax+1

    for ii in range(1, Np):
        g = zgr_h[idx_pred[ii]]
        emit = np.zeros(G) if np.isnan(g) else (g - zref_grid) ** 2
        # min_{k} prev[shift] + pen[k]  -> 窗口卷积取 argmin
        best = np.full(G, np.inf)
        arg = np.zeros(G, dtype=np.int32)
        for ki, k in enumerate(ks):
            shifted = np.full(G, np.inf)
            if k >= 0:
                shifted[k:] = prev[:G - k] if k > 0 else prev
            else:
                shifted[:G + k] = prev[-k:]
            cand = shifted + pen[ki]
            upd = cand < best
            best[upd] = cand[upd]
            # 来源网格 = 当前 g 索引 - k
            src = np.arange(G) - k
            arg[upd] = src[upd]
        prev = best + emit
        back[ii] = arg

    # 回溯
    path = np.empty(Np, dtype=np.int32)
    path[-1] = int(np.argmin(prev))
    for ii in range(Np - 1, 0, -1):
        path[ii - 1] = back[ii, path[ii]]
    out[ps:] = grid[path]
    return out


# ---------- 评估 ----------
DATA = os.path.join(os.path.dirname(__file__), "data", "extracted")
def _load(folder, wid):
    h = pd.read_csv(os.path.join(folder, f"{wid}__horizontal_well.csv"))
    tw = glob.glob(os.path.join(folder, f"{wid}__typewell*.csv"))
    t = pd.read_csv(tw[0])
    return h, t

def _ids(folder):
    return sorted(os.path.basename(f).split("__")[0]
                  for f in glob.glob(os.path.join(folder, "*__horizontal_well.csv")))


def cv(lam=1.0, n=25, seed=0, **kw):
    import random
    ids = _ids(os.path.join(DATA, "train"))
    random.seed(seed); samp = random.sample(ids, n)
    err_dp, err_const, npts = [], [], []
    for wid in samp:
        h, t = _load(os.path.join(DATA, "train"), wid)
        ps = h["TVT_input"].notna().sum()
        if ps < 20 or len(h) - ps < 10:
            continue
        truth = h["TVT"].values[ps:]
        pred = align_well(h, t, lam=lam, **kw)[ps:]
        err_dp.append(pred - truth)
        err_const.append(np.full(len(truth), h["TVT_input"].values[ps-1]) - truth)
        npts.append(len(truth))
    e_dp = np.concatenate(err_dp); e_c = np.concatenate(err_const)
    return (np.sqrt(np.mean(e_dp**2)), np.sqrt(np.mean(e_c**2)), len(npts))


if __name__ == "__main__":
    import sys, time
    print("=== λ 扫描 (25 井, pooled RMSE) ===")
    for lam in [0.1, 0.3, 1.0, 3.0, 10.0]:
        t0 = time.time()
        dp, const, k = cv(lam=lam, n=25)
        print(f"  lam={lam:<5}  DP={dp:7.3f}  const={const:7.3f}  ({k}井, {time.time()-t0:.1f}s)")
