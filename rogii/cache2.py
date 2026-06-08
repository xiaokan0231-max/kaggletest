"""
增强缓存: 每口井存 selector预测 + 真值 + 常数 + 每点集成分歧度 + beam + pf_mean。
用于测试"高分歧->回退"的自适应融合(挽救6%灾难井)。
"""
import heur, numpy as np, pickle, os, time
from joblib import Parallel, delayed

CACHE2 = os.path.join(os.path.dirname(__file__), "selcache2.pkl")


def _enrich_one(wid, n_seeds, min_ps, min_ev):
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
    # 一次跑出 raw 集成
    pred_arr, liks = heur.run_pf_raw(hw, tw, n_seeds=n_seeds)
    pf_by_scale = heur.pf_by_scale_from_raw(pred_arr, liks)
    # selector 融合
    code, variant, _, _ = heur.selector_well_code(hw)
    beam_full = heur.run_beam_ensemble(hw, tw)
    kn = hw['TVT_input'].dropna()
    last = float(kn.iloc[-1]) if len(kn) else float(np.nanmean(beam_full))
    sel_full = heur.apply_selector_variant(variant, pf_by_scale, beam_full, last)
    # 每点分歧度 = 跨种子 std
    disagree_full = np.std(pred_arr, axis=0)
    md = hw['MD'].values.astype(float)
    md_since = np.maximum(md[ps:] - md[ps - 1], 0.0)
    return dict(wid=wid, ps=ps,
                pred=sel_full[ps:].astype(float),
                truth=truth, last=last, md_since=md_since,
                disagree=disagree_full[ps:].astype(float),
                beam=beam_full[ps:].astype(float),
                pf_mean=pf_by_scale['pf_mean'][ps:].astype(float),
                variant=variant)


def build(n=None, n_seeds=128, n_jobs=-1, min_ps=60, min_ev=30):
    ids = heur.well_ids('train')
    if n:
        ids = ids[:n]
    t0 = time.time()
    res = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_enrich_one)(w, n_seeds, min_ps, min_ev) for w in ids)
    res = [r for r in res if r is not None]
    pickle.dump(res, open(CACHE2, 'wb'))
    print(f"enriched cache: {len(res)} wells, {time.time()-t0:.1f}s -> {CACHE2}", flush=True)
    return res


def load():
    return pickle.load(open(CACHE2, 'rb'))


def pooled(cache, fn=None):
    """fn(d)->处理后预测; None=原始selector。返回全井 pooled RMSE。"""
    errs = [(d['pred'] if fn is None else fn(d)) - d['truth'] for d in cache]
    e = np.concatenate(errs)
    return float(np.sqrt(np.mean(e**2)))


if __name__ == "__main__":
    import sys
    if not os.path.exists(CACHE2) or (len(sys.argv) > 1 and sys.argv[1] == 'rebuild'):
        build(n_seeds=128)
    cache = load()
    base = pooled(cache)
    n_pts = sum(len(d['truth']) for d in cache)
    print(f"增强缓存: {len(cache)}井 {n_pts}点  baseline selector pooled RMSE = {base:.4f}")

    # 分歧度统计
    alldis = np.concatenate([d['disagree'] for d in cache])
    print(f"分歧度: 中位={np.median(alldis):.2f} p90={np.percentile(alldis,90):.2f} p99={np.percentile(alldis,99):.2f}")

    # 自适应融合: w = clip((disagree-d0)/dscale, 0, wmax), pred=(1-w)*sel + w*target
    def make_adapt(target_key, d0, dscale, wmax):
        def fn(d):
            w = np.clip((d['disagree'] - d0) / dscale, 0.0, wmax)
            tgt = d['last'] if target_key == 'const' else d[target_key]
            return (1 - w) * d['pred'] + w * tgt
        return fn

    print("\n--- 自适应回退 const (高分歧->拉向常数) ---")
    for d0 in [4, 6, 8]:
        for dscale in [6, 10]:
            for wmax in [0.4, 0.7, 1.0]:
                r = pooled(cache, make_adapt('const', d0, dscale, wmax))
                tag = "  <==" if r < base - 0.01 else ""
                print(f"  const d0={d0} ds={dscale} wmax={wmax} = {r:.4f} (Δ{r-base:+.4f}){tag}")
    print("\n--- 自适应回退 beam ---")
    for d0 in [4, 6, 8]:
        for dscale in [6, 10]:
            for wmax in [0.4, 0.7, 1.0]:
                r = pooled(cache, make_adapt('beam', d0, dscale, wmax))
                tag = "  <==" if r < base - 0.01 else ""
                print(f"  beam d0={d0} ds={dscale} wmax={wmax} = {r:.4f} (Δ{r-base:+.4f}){tag}")
