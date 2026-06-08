"""
本地复现公开方案的"启发式分支"(PF 似然集成 + Beam 集成 + selector 融合),
在训练井上做留出 CV(模拟无泄漏: 只用 TVT_input 已知段 + GR, 预测 PS 后).
纯 numpy, 不需要 numba/koolbox/catboost。目标: 提升无泄漏井上的真实 RMSE。
"""
import numpy as np, pandas as pd, glob, os, time
from scipy.signal import savgol_filter

DATA = os.path.join(os.path.dirname(__file__), "data", "extracted")

# ---------------- selector / 集成 配置 (来自公开方案 CELL 6) ----------------
SELECTOR_N_EVAL_THRESHOLD = 4840.0
SELECTOR_Z_SPAN_THRESHOLDS = (136.73000000000016, 185.5133333333342)
SELECTOR_BIN_VARIANTS = {
    0: 'pf_scale_5_hold_0.2',
    1: 'pf_scale_3_hold_0.15',
    2: 'pf_scale_12_beam_0.2_hold_0.15',
    3: 'pf_scale_5_hold_0.15',
    4: 'pf_scale_5_beam_0.05_hold_0.05',
    5: 'pf_scale_12_beam_0.2_hold_0.05',
}
SELECTOR_GLOBAL_VARIANT = 'pf_scale_8_hold_0.2'
SELECTOR_SCALES = (3.0, 5.0, 8.0, 12.0)
BEAM_CONFIGS = [
    (10, 20.0, 144.0, 2), (10, 8.0, 64.0, 2), (8, 35.0, 220.0, 1),
    (10, 14.0, 90.0, 5), (20, 4.0, 36.0, 3), (12, 12.0, 100.0, 3),
    (15, 25.0, 180.0, 2), (20, 30.0, 200.0, 2), (15, 10.0, 80.0, 4),
    (25, 6.0, 50.0, 3), (10, 40.0, 300.0, 1), (12, 18.0, 120.0, 5),
    (30, 8.0, 70.0, 2), (10, 50.0, 400.0, 0),
]


def load_well(wid, split='train'):
    base = os.path.join(DATA, split)
    hw = pd.read_csv(os.path.join(base, f'{wid}__horizontal_well.csv'))
    tw = pd.read_csv(os.path.join(base, f'{wid}__typewell.csv'))
    return hw, tw


def tvt_from_contacts(hw_tr, tw_tr, ref_col='EGFDU'):
    """物理模型(可见井泄漏): 用 train 版的 EGFDU 接触面 + TVT 均值偏移反推。返回 Series。"""
    tw_g = tw_tr.dropna(subset=['Geology'])
    ref_tvt = tw_g[tw_g['Geology'] == ref_col]['TVT'].min()
    if np.isnan(ref_tvt):
        ref_col = tw_g['Geology'].iloc[0]
        ref_tvt = tw_g[tw_g['Geology'] == ref_col]['TVT'].min()
    offset = (hw_tr['TVT'] - (ref_tvt - (hw_tr['Z'] - hw_tr[ref_col]))).mean()
    return ref_tvt - (hw_tr['Z'] - hw_tr[ref_col]) + offset


def run_particle_filter(hw, tw, n_particles=500, seed=42,
                        mom=0.998, vn=0.002, pn=0.005, rp=0.1, rr=0.001,
                        resamp=0.5, init_pos=4.5, init_rate=0.01):
    tw_s = tw.sort_values('TVT')
    tw_tvt = tw_s['TVT'].values.astype(float)
    tw_gr = tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)
    kn = hw[hw['TVT_input'].notna()]
    ev = hw[hw['TVT_input'].isna()]
    if len(ev) == 0:
        return hw['TVT_input'].values.astype(float).copy(), 0.0
    last = kn.iloc[-1]
    last_tvt = float(last['TVT_input']); last_Z = float(last['Z']); last_MD = float(last['MD'])
    tw_at_k = np.interp(kn['TVT_input'].values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn['GR'].fillna(0).values - tw_at_k), 10., 60.))
    tail = kn.tail(30)
    dt = np.diff(tail['TVT_input'].values); dz = np.diff(tail['Z'].values); dm = np.diff(tail['MD'].values)
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0
    N = n_particles
    rng = np.random.default_rng(seed)
    ls = last_tvt + last_Z
    pos = ls + init_pos * rng.standard_normal(N)
    rate = ir + init_rate * rng.standard_normal(N)
    w = np.ones(N) / N
    MOM, VN, PN, RP, RR, RESAMP = mom, vn, pn, rp, rr, resamp
    md_v = ev['MD'].values.astype(float); z_v = ev['Z'].values.astype(float)
    gr_interp = hw['GR'].interpolate(limit_direction='both').fillna(tw_gr.mean())
    gr_v = gr_interp.values.astype(float)[ev.index]
    out_vals = hw['TVT_input'].values.astype(float).copy()
    res = np.empty(len(ev)); prev_MD = last_MD; log_lik = 0.0
    for i in range(len(ev)):
        dm_step = max(md_v[i] - prev_MD, 1.0)
        rate = MOM * rate + VN * rng.standard_normal(N)
        pos = pos + rate * dm_step + PN * rng.standard_normal(N)
        tvt_p = pos - z_v[i]
        tvt_p = np.clip(tvt_p, tw_tvt[0] - 100, tw_tvt[-1] + 100)
        pos = tvt_p + z_v[i]
        eg = np.interp(tvt_p, tw_tvt, tw_gr)
        d = (gr_v[i] - eg) / gs
        lk = np.exp(-0.5 * np.minimum(d**2, 600.))
        lk = np.maximum(lk, 1e-300)
        avg_lk = float((w * lk).sum())
        log_lik += np.log(max(avg_lk, 1e-300))
        w = w * lk; ws = w.sum()
        w = w / ws if ws > 0 else np.ones(N) / N
        n_eff = 1.0 / (w**2).sum()
        if n_eff < RESAMP * N:
            cum = np.cumsum(w); u0 = rng.uniform(0, 1.0 / N)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
            pos = pos[idx] + RP * rng.standard_normal(N)
            rate = rate[idx] + RR * rng.standard_normal(N)
            w = np.ones(N) / N
        res[i] = float(np.dot(w, pos - z_v[i]))
        prev_MD = md_v[i]
    out_vals[list(ev.index)] = res
    return out_vals, log_lik


def run_pf_lik_ensemble_scales(hw, tw, scales=SELECTOR_SCALES, n_particles=500, n_seeds=128, pf_kw=None):
    pf_kw = pf_kw or {}
    preds = []; liks = []
    for s in range(n_seeds):
        p, ll = run_particle_filter(hw, tw, n_particles=n_particles, seed=s, **pf_kw)
        preds.append(p); liks.append(ll)
    pred_arr = np.stack(preds, 0)
    liks = np.array(liks); liks_n = liks - liks.max()
    out = {}
    for scale in scales:
        weights = np.exp(liks_n / float(scale)); weights /= weights.sum()
        out[f'pf_scale_{scale:g}'] = (weights[:, None] * pred_arr).sum(0)
    out['pf_mean'] = pred_arr.mean(0)
    return out


def beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs=10, mc=20.0, es=144.0, r=2):
    n = len(hgr); nt = len(tw_tvt)
    if n == 0:
        return np.array([last_tvt])
    if r > 0 and n > max(3, 2 * r + 1):
        win = min(2 * r + 1, n if n % 2 == 1 else n - 1)
        sgr = savgol_filter(hgr, win, min(2, win - 1))
    else:
        sgr = hgr.copy()
    si = int(np.argmin(np.abs(tw_tvt - last_tvt)))
    MOVES = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
    MC = mc * np.array([2., 1., 0., 1., 2.])
    bidx = np.full(bs, si, dtype=np.int64); bcost = np.full(bs, np.inf); bcost[0] = 0.; bn = 1
    result = np.zeros(n)
    for step in range(n):
        gv = sgr[step]
        ni = bidx[:bn, None] + MOVES[None, :]
        ci = np.clip(ni, 0, nt - 1)
        valid = (ni >= 0) & (ni < nt)
        gr_e = (gv - tw_gr[ci])**2 / es
        tot = bcost[:bn, None] + gr_e + MC[None, :]
        tot = np.where(valid, tot, np.inf)
        ni_f = ni.flatten(); tot_f = tot.flatten(); vf = valid.flatten()
        ni_f = ni_f[vf]; tot_f = tot_f[vf]
        order = np.argsort(tot_f); ni_s = ni_f[order]; tot_s = tot_f[order]
        _, first = np.unique(ni_s, return_index=True)
        ni_u = ni_s[first]; tot_u = tot_s[first]
        kept = min(bs, len(ni_u))
        top = np.argpartition(tot_u, min(kept - 1, len(tot_u) - 1))[:kept]
        top = top[np.argsort(tot_u[top])]
        bidx[:kept] = ni_u[top]; bcost[:kept] = tot_u[top]
        if kept < bs:
            bidx[kept:] = bidx[kept - 1]; bcost[kept:] = np.inf
        bn = kept
        result[step] = tw_tvt[bidx[0]]
    return result


def run_beam_ensemble(hw, tw, configs=BEAM_CONFIGS):
    kn = hw[hw['TVT_input'].notna()]; ev = hw[hw['TVT_input'].isna()]
    if len(ev) == 0:
        return hw['TVT_input'].values.astype(float).copy()
    last_tvt = float(kn.iloc[-1]['TVT_input'])
    tw_s = tw.sort_values('TVT')
    tw_tvt = tw_s['TVT'].values.astype(float)
    tw_gr = tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)
    gr_all = hw['GR'].interpolate(limit_direction='both').fillna(tw_gr.mean()).values.astype(float)
    hgr = gr_all[ev.index]
    beam_results = [beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r) for (bs, mc, es, r) in configs]
    beam_mean = np.stack(beam_results, 0).mean(0)
    out = hw['TVT_input'].values.astype(float).copy()
    out[list(ev.index)] = beam_mean
    return out


def selector_well_code(hw):
    eval_mask = hw['TVT_input'].isna().to_numpy()
    n_eval = float(eval_mask.sum())
    z_eval = hw.loc[eval_mask, 'Z'].values.astype(float)
    z_span = float(np.nanmax(z_eval) - np.nanmin(z_eval)) if len(z_eval) else 0.0
    n_bin = int(n_eval > SELECTOR_N_EVAL_THRESHOLD)
    z_bin = int(np.searchsorted(SELECTOR_Z_SPAN_THRESHOLDS, z_span, side='right'))
    code = n_bin + 2 * z_bin
    variant = SELECTOR_BIN_VARIANTS.get(code, SELECTOR_GLOBAL_VARIANT)
    return code, variant, n_eval, z_span


def parse_selector_variant(name):
    parts = name.split('_')
    scale = float(parts[2]); beam_weight = 0.0; hold_weight = 0.0
    if 'beam' in parts:
        beam_weight = float(parts[parts.index('beam') + 1])
    if 'hold' in parts:
        hold_weight = float(parts[parts.index('hold') + 1])
    return scale, beam_weight, hold_weight


def apply_selector_variant(name, pf_by_scale, tvt_beam, last_known_tvt):
    scale, beam_weight, hold_weight = parse_selector_variant(name)
    base = pf_by_scale.get(f'pf_scale_{scale:g}')
    if base is None:
        base = pf_by_scale[SELECTOR_GLOBAL_VARIANT.split('_beam_')[0].split('_hold_')[0]]
    pred = (1.0 - beam_weight) * base + beam_weight * tvt_beam
    pred = (1.0 - hold_weight) * pred + hold_weight * last_known_tvt
    return pred


def selector_predict(hw, tw, n_seeds=128, pf_kw=None):
    """无泄漏启发式分支: 完整复现 selector 融合 (PF集成 + Beam + variant融合)。返回整段 TVT 预测。"""
    code, variant, n_eval, z_span = selector_well_code(hw)
    pf_by_scale = run_pf_lik_ensemble_scales(hw, tw, n_seeds=n_seeds, pf_kw=pf_kw)
    tvt_beam = run_beam_ensemble(hw, tw)
    kn = hw['TVT_input'].dropna()
    last_known_tvt = float(kn.iloc[-1]) if len(kn) else float(np.nanmean(tvt_beam))
    return apply_selector_variant(variant, pf_by_scale, tvt_beam, last_known_tvt)


# ---------------- 改进候选 (模块级, 可并行 pickle) ----------------
def _ps_of(hw):
    return int(hw['TVT_input'].notna().sum())


def selector_smooth(hw, tw, n_seeds=128, sg_w=31, sg_p=3, pf_kw=None):
    """基线 selector 之上对预测段做 Savgol 平滑。"""
    pred = selector_predict(hw, tw, n_seeds=n_seeds, pf_kw=pf_kw)
    ps = _ps_of(hw)
    seg = pred[ps:].astype(float)
    if len(seg) >= sg_p + 2:
        wl = min(sg_w, len(seg)); wl = wl if wl % 2 == 1 else wl - 1
        if wl >= sg_p + 2:
            seg = savgol_filter(seg, wl, sg_p)
    out = pred.copy(); out[ps:] = seg
    return out


def selector_hold(hw, tw, n_seeds=128, extra_hold=0.1, pf_kw=None):
    """额外把预测段拉回 last_known(常数) extra_hold 比例。"""
    pred = selector_predict(hw, tw, n_seeds=n_seeds, pf_kw=pf_kw)
    ps = _ps_of(hw)
    kn = hw['TVT_input'].dropna()
    last = float(kn.iloc[-1]) if len(kn) else float(np.nanmean(pred))
    out = pred.copy()
    out[ps:] = (1 - extra_hold) * pred[ps:] + extra_hold * last
    return out


def selector_ramp(hw, tw, n_seeds=128, tau=400.0, pf_kw=None):
    """按 md_since 自适应: 近 PS 偏常数, 远处偏 PF。ramp=1-exp(-md_since/tau)。"""
    pred = selector_predict(hw, tw, n_seeds=n_seeds, pf_kw=pf_kw)
    ps = _ps_of(hw)
    kn = hw['TVT_input'].dropna()
    last = float(kn.iloc[-1]) if len(kn) else float(np.nanmean(pred))
    md = hw['MD'].values.astype(float)
    md_ps = md[ps - 1] if ps > 0 else md[0]
    md_since = np.maximum(md[ps:] - md_ps, 0.0)
    ramp = 1.0 - np.exp(-md_since / tau)
    out = pred.copy()
    out[ps:] = last + ramp * (pred[ps:] - last)
    return out


def selector_revhold(hw, tw, n_seeds=128, sg_w=51, sg_p=3, hmax=0.15, tau=600.0, pf_kw=None):
    """递增 hold: 近PS不拉, 远处按 md_since 递增拉回常数(漂移最严重处多约束)。含平滑。"""
    pred = selector_smooth(hw, tw, n_seeds=n_seeds, sg_w=sg_w, sg_p=sg_p, pf_kw=pf_kw)
    ps = _ps_of(hw)
    kn = hw['TVT_input'].dropna()
    last = float(kn.iloc[-1]) if len(kn) else float(np.nanmean(pred))
    md = hw['MD'].values.astype(float)
    md_ps = md[ps - 1] if ps > 0 else md[0]
    md_since = np.maximum(md[ps:] - md_ps, 0.0)
    hold_w = hmax * (1.0 - np.exp(-md_since / tau))
    out = pred.copy()
    out[ps:] = (1.0 - hold_w) * pred[ps:] + hold_w * last
    return out


def selector_smooth_hold(hw, tw, n_seeds=128, sg_w=31, sg_p=3, extra_hold=0.08, pf_kw=None):
    pred = selector_smooth(hw, tw, n_seeds=n_seeds, sg_w=sg_w, sg_p=sg_p, pf_kw=pf_kw)
    ps = _ps_of(hw)
    kn = hw['TVT_input'].dropna()
    last = float(kn.iloc[-1]) if len(kn) else float(np.nanmean(pred))
    out = pred.copy()
    out[ps:] = (1 - extra_hold) * pred[ps:] + extra_hold * last
    return out


# ---------------- CV ----------------
def well_ids(split='train'):
    return sorted(os.path.basename(f).split('__')[0]
                  for f in glob.glob(os.path.join(DATA, split, '*__horizontal_well.csv')))


def cv(n=40, seed=0, n_seeds=64, predictor=None, verbose=True, min_ps=60, min_ev=30):
    """对采样训练井评估启发式分支 vs 常数基线。pooled RMSE on post-PS。"""
    import random
    predictor = predictor or (lambda hw, tw: selector_predict(hw, tw, n_seeds=n_seeds))
    ids = well_ids('train')
    random.seed(seed); samp = random.sample(ids, min(n, len(ids)))
    e_sel, e_const = [], []
    imp = 0; used = 0; t0 = time.time()
    for wid in samp:
        try:
            hw, tw = load_well(wid, 'train')
        except Exception:
            continue
        if 'TVT' not in hw.columns:
            continue
        ps = int(hw['TVT_input'].notna().sum())
        n_ev = len(hw) - ps
        if ps < min_ps or n_ev < min_ev:
            continue
        truth = hw['TVT'].values[ps:].astype(float)
        if np.isnan(truth).any():
            continue
        pred = predictor(hw, tw)[ps:]
        const = float(hw['TVT_input'].values[ps - 1])
        rsel = np.sqrt(np.mean((pred - truth)**2))
        rconst = np.sqrt(np.mean((const - truth)**2))
        e_sel.append(pred - truth); e_const.append(const - truth)
        imp += (rsel < rconst); used += 1
    e_sel = np.concatenate(e_sel); e_const = np.concatenate(e_const)
    r = dict(sel=float(np.sqrt(np.mean(e_sel**2))),
             const=float(np.sqrt(np.mean(e_const**2))),
             wells=used, better=imp, secs=time.time() - t0)
    if verbose:
        print(f"selector={r['sel']:.4f}  const={r['const']:.4f}  "
              f"改善井={r['better']}/{r['wells']}  {r['secs']:.1f}s")
    return r


def _eval_one(wid, pred_fn, pred_kw, min_ps, min_ev):
    """单井评估, 返回 (err_sel, err_const) 或 None。pred_fn 须为模块级函数(可pickle)。"""
    try:
        hw, tw = load_well(wid, 'train')
    except Exception:
        return None
    if 'TVT' not in hw.columns:
        return None
    ps = int(hw['TVT_input'].notna().sum())
    n_ev = len(hw) - ps
    if ps < min_ps or n_ev < min_ev:
        return None
    truth = hw['TVT'].values[ps:].astype(float)
    if np.isnan(truth).any():
        return None
    pred = pred_fn(hw, tw, **pred_kw)[ps:]
    const = float(hw['TVT_input'].values[ps - 1])
    return (pred - truth, np.full_like(truth, const) - truth)


def cv_par(n=40, seed=0, pred_fn=None, pred_kw=None, n_jobs=-1, min_ps=60, min_ev=30, verbose=True):
    """并行版 CV。pred_fn(hw, tw, **pred_kw)->整段TVT预测; 须为模块级函数。"""
    import random
    from joblib import Parallel, delayed
    pred_fn = pred_fn or selector_predict
    pred_kw = pred_kw or {}
    ids = well_ids('train')
    random.seed(seed); samp = random.sample(ids, min(n, len(ids)))
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_eval_one)(wid, pred_fn, pred_kw, min_ps, min_ev) for wid in samp)
    results = [r for r in results if r is not None]
    e_sel = np.concatenate([r[0] for r in results])
    e_const = np.concatenate([r[1] for r in results])
    imp = sum(np.sqrt(np.mean(r[0]**2)) < np.sqrt(np.mean(r[1]**2)) for r in results)
    out = dict(sel=float(np.sqrt(np.mean(e_sel**2))),
               const=float(np.sqrt(np.mean(e_const**2))),
               wells=len(results), better=imp, secs=time.time() - t0)
    if verbose:
        print(f"selector={out['sel']:.4f}  const={out['const']:.4f}  "
              f"改善井={out['better']}/{out['wells']}  {out['secs']:.1f}s")
    return out


if __name__ == "__main__":
    import sys
    # 单井计时
    ids = well_ids('train')
    hw, tw = load_well(ids[0], 'train')
    ps = int(hw['TVT_input'].notna().sum())
    print(f"sample well {ids[0]}: rows={len(hw)} ps={ps} ev={len(hw)-ps}")
    for ns in [16, 64]:
        t0 = time.time()
        p = selector_predict(hw, tw, n_seeds=ns)
        truth = hw['TVT'].values[ps:].astype(float)
        rmse = np.sqrt(np.mean((p[ps:] - truth)**2))
        print(f"  n_seeds={ns}: rmse={rmse:.3f}  {time.time()-t0:.2f}s")
