"""
移植 top-3 方案的 run_pf_z (Z-速度耦合 PF): 用井眼 Z 轨迹学到的 dTVT~dZ 线性关系
约束粒子速度 + 点GR/平滑GR 双似然。对比 selector 简单点-GR PF 基准。
纯 numpy/scipy, 可本地并行 CV。
"""
import heur, numpy as np, pandas as pd, time
from scipy.interpolate import interp1d

# 常数 (来自 top-3)
PF_MOM=0.993; PF_VN=0.005; PF_PN=0.01
PF_GR_SIG_MIN=10.; PF_GR_SIG_MAX=60.; PF_GR_SIG_DEF=30.
PF_INIT_V_STD=0.02; PF_INIT_SPR=0.5; PF_RESAMP=0.5
PF_ROUGH_P=0.2; PF_ROUGH_V=0.003; PF_GR_WIN=5; PF_GR_WT=0.3


def _gr_sigma(hw, tw_tvt, tw_gr):
    kn = hw[hw['TVT_input'].notna() & hw['GR'].notna()]
    if len(kn) < 20:
        return PF_GR_SIG_DEF
    return float(np.clip(np.std(kn['GR'].values - np.interp(kn['TVT_input'].values, tw_tvt, tw_gr)),
                         PF_GR_SIG_MIN, PF_GR_SIG_MAX))


def run_pf_z(hw, tw_tvt, tw_gr, N=500, seed=0):
    rng = np.random.default_rng(seed)
    tw_s = pd.Series(tw_gr).rolling(PF_GR_WIN, center=True, min_periods=1).mean().values
    tf_p = interp1d(tw_tvt, tw_gr, bounds_error=False, fill_value=(tw_gr[0], tw_gr[-1]))
    tf_s = interp1d(tw_tvt, tw_s, bounds_error=False, fill_value=(tw_s[0], tw_s[-1]))
    tmin, tmax = tw_tvt.min(), tw_tvt.max()
    gs = _gr_sigma(hw, tw_tvt, tw_gr)
    kna = hw[hw['TVT_input'].notna()]; ev = hw[hw['TVT_input'].isna()]
    out = hw['TVT_input'].values.astype(float).copy()
    if len(ev) == 0:
        return out, 0.0
    # 学 dTVT ~ beta*dZ + icpt
    dz_k = np.diff(kna['Z'].values); dvt = np.diff(kna['TVT_input'].values)
    dmd_k = np.diff(kna['MD'].values); m2 = dmd_k > 0
    if m2.sum() >= 10:
        vz = dz_k[m2] / dmd_k[m2]; vt = dvt[m2] / dmd_k[m2]
        A = np.column_stack([vz, np.ones_like(vz)])
        c, _, _, _ = np.linalg.lstsq(A, vt, rcond=None)
        beta, icpt, zsig = c[0], c[1], max(np.std(vt - (c[0]*vz + c[1])), 0.001)
    else:
        beta, icpt, zsig = -1., 0., 0.1
    tail2 = kna.tail(20); dvt2 = np.diff(tail2['TVT_input'].values)
    dmd2 = np.diff(tail2['MD'].values); m3 = dmd2 > 0
    iv = float(np.median(dvt2[m3] / dmd2[m3])) if m3.sum() >= 3 else 0.
    gr_sm = hw['GR'].rolling(PF_GR_WIN, center=True, min_periods=1).mean().values
    pos = float(kna['TVT_input'].iloc[-1]) + rng.normal(0., PF_INIT_SPR, N)
    vel = iv + rng.normal(0., PF_INIT_V_STD, N); w = np.ones(N) / N
    ev_pos = np.where(hw['TVT_input'].isna().values)[0]
    md_v = ev['MD'].values; gr_v = ev['GR'].values; z_v = ev['Z'].values
    gr_sm_ev = gr_sm[ev_pos]
    pm = float(kna['MD'].iloc[-1]); pz = float(kna['Z'].iloc[-1])
    pts = np.empty(len(ev)); log_lik = 0.0
    for i in range(len(ev)):
        dm = max(md_v[i] - pm, 1.); dzd = (z_v[i] - pz) / dm; ve = beta*dzd + icpt
        vel = PF_MOM*vel + rng.normal(0., PF_VN, N)
        pos = pos + vel*dm + rng.normal(0., PF_PN, N)
        pos = np.clip(pos, tmin-50., tmax+50.)
        if not np.isnan(gr_v[i]):
            ep = tf_p(pos); lp = np.exp(-0.5*((gr_v[i]-ep)/gs)**2)
            gsm = gr_sm_ev[i]
            if not np.isnan(gsm):
                ls2 = np.exp(-0.5*((gsm - tf_s(pos))/(gs*1.5))**2)
                lk = (1-PF_GR_WT)*lp + PF_GR_WT*ls2
            else:
                lk = lp
            lk = np.maximum(lk, 1e-300)
            log_lik += np.log(max(float((w*lk).sum()), 1e-300))
            w *= lk; ws = w.sum(); w = (w/ws) if ws > 0 else np.full(N, 1./N)
        lz = np.exp(-0.5*((vel-ve)/max(zsig*2., 0.005))**2)
        lz = np.maximum(lz, 1e-300); w *= lz; ws = w.sum()
        w = (w/ws) if ws > 0 else np.full(N, 1./N)
        ne = 1./np.sum(w**2)
        if ne < PF_RESAMP*N:
            ix = np.searchsorted(np.cumsum(w), (np.arange(N)+rng.uniform())/N)
            ix = np.clip(ix, 0, N-1); pos = pos[ix]; vel = vel[ix]; w[:] = 1./N
            pos += rng.normal(0., PF_ROUGH_P, N); vel += rng.normal(0., PF_ROUGH_V, N)
        pts[i] = np.average(pos, weights=w)
        pm = md_v[i]; pz = z_v[i]
    out[ev_pos] = pts
    return out, log_lik


def pfz_predict(hw, tw, n_seeds=32, scale=8.0):
    """Z-耦合 PF 的似然加权集成 (模仿 selector 的 scale 加权)。"""
    tw_s = tw.sort_values('TVT')
    tw_tvt = tw_s['TVT'].values.astype(float)
    tw_gr = tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)
    preds = []; liks = []
    for s in range(n_seeds):
        p, ll = run_pf_z(hw, tw_tvt, tw_gr, seed=s)
        preds.append(p); liks.append(ll)
    arr = np.stack(preds, 0); liks = np.array(liks)
    liks_n = liks - liks.max(); ww = np.exp(liks_n/scale); ww /= ww.sum()
    return (ww[:, None]*arr).sum(0)


def pfz_mean(hw, tw, n_seeds=32):
    return pfz_predict(hw, tw, n_seeds=n_seeds, scale=1e9)  # 近似简单均值


if __name__ == "__main__":
    import heur
    print("对比: selector基准 vs run_pf_z (Z耦合), 150井")
    r0 = heur.cv_par(n=150, seed=7, pred_fn=heur.selector_predict, pred_kw={'n_seeds':32}, verbose=False)
    print(f"  selector(简单PF)   = {r0['sel']:.4f}")
    r1 = heur.cv_par(n=150, seed=7, pred_fn=pfz_predict, pred_kw={'n_seeds':32,'scale':8.0}, verbose=False)
    print(f"  run_pf_z scale=8   = {r1['sel']:.4f}  (Δ {r1['sel']-r0['sel']:+.4f})")
    r2 = heur.cv_par(n=150, seed=7, pred_fn=pfz_mean, pred_kw={'n_seeds':32}, verbose=False)
    print(f"  run_pf_z mean      = {r2['sel']:.4f}  (Δ {r2['sel']-r0['sel']:+.4f})")
