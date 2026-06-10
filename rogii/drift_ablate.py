"""
净漂移回归的消融+稳健性鞭刑。
风险拆解: 全特征版叠加了v4(训练模型)+v6(空间特征位置依赖)两种翻车模式。
消融问题: 去掉空间特征后还剩多少? 纯自井特征版transfer风险最低。
鞭刑: ①分半泛化(4种子) ②空间分块CV(模拟位置偏移, 这次用对闸门)。
"""
import numpy as np, pickle, pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
import heur
from joblib import Parallel, delayed
import time

sc = {d['wid']: d for d in pickle.load(open('selcache.pkl', 'rb'))}

# 复用 drift_reg 的特征构建(直接import函数太繁, 重算一次并缓存)
import importlib.util
spec = importlib.util.spec_from_file_location("drift_reg_mod", "drift_reg.py")

# 改为直接重建特征表(同drift_reg逻辑, 但缓存到pkl)
import os
if os.path.exists('drift_feats.pkl'):
    df = pd.read_pickle('drift_feats.pkl')
    print(f"载入缓存特征表 {df.shape}", flush=True)
else:
    dg = {r['wid']: r for r in pickle.load(open('distgate.pkl', 'rb'))}
    c2 = {d['wid']: d for d in pickle.load(open('selcache2.pkl', 'rb'))}
    def feats(wid):
        if wid not in sc: return None
        try: hw, tw = heur.load_well(wid, 'train')
        except Exception: return None
        d = sc[wid]
        target = float(np.mean(d['pred'] - d['truth']))
        ps = int(hw['TVT_input'].notna().sum())
        kn = hw.iloc[:ps]; ev = hw.iloc[ps:]
        md = hw['MD'].values; z = hw['Z'].values
        twt = tw.sort_values('TVT'); tw_tvt = twt['TVT'].values
        tw_gr = twt['GR'].fillna(twt['GR'].mean()).values
        twg = tw['GR'].dropna().values
        tail = kn.tail(40); sl = np.polyfit(tail['MD'], tail['TVT_input'], 1)[0] if len(tail)>5 else 0.
        tail2 = kn.tail(120); sl2 = np.polyfit(tail2['MD'], tail2['TVT_input'], 1)[0] if len(tail2)>10 else 0.
        p = d['pred']
        gr_at_k = np.interp(kn['TVT_input'].values, tw_tvt, tw_gr)
        f = dict(wid=wid, target=target,
                 tail_slope=sl, tail_slope_long=sl2, slope_diff=sl-sl2,
                 pred_slope=np.polyfit(np.arange(len(p)), p, 1)[0] if len(p)>10 else 0.,
                 pred_net=float(p[-1]-p[0]),
                 ps_frac=ps/len(hw), n_ev=len(ev), md_span_ev=md[-1]-md[ps-1],
                 z_net_ev=float(z[-1]-z[ps-1]), z_span_ev=float(np.ptp(z[ps:])),
                 gr_mis=float(np.nanmean(kn['GR'].values - gr_at_k)),
                 gr_ev_m=float(np.nanmean(ev['GR'])), gr_kn_m=float(np.nanmean(kn['GR'])),
                 gr_ev_minus_kn=float(np.nanmean(ev['GR'])-np.nanmean(kn['GR'])),
                 twg_m=float(twg.mean()), twg_rng=float(np.ptp(twg)),
                 tvt_last=float(kn['TVT_input'].iloc[-1]),
                 tvt_pos_in_tw=float((kn['TVT_input'].iloc[-1]-tw_tvt.min())/max(np.ptp(tw_tvt),1e-9)),
                 x_mean=float(hw['X'].mean()), y_mean=float(hw['Y'].mean()))
        if wid in dg:
            r = dg[wid]
            f['sp_minus_sel'] = float(np.mean(r['sp_err']) - np.mean(d['pred']-d['truth']))
            f['nn_min'] = r['nn_min']; f['nn_med'] = r['nn_med']
        else:
            f['sp_minus_sel']=0.; f['nn_min']=9999.; f['nn_med']=9999.
        if wid in c2:
            f['disagree']=float(np.mean(c2[wid]['disagree']))
            f['beam_minus_sel']=float(np.mean(c2[wid]['beam']-c2[wid]['pred']))
        else:
            f['disagree']=np.nan; f['beam_minus_sel']=np.nan
        return f
    t0=time.time()
    rows=[r for r in Parallel(n_jobs=-1)(delayed(feats)(w) for w in sc.keys()) if r]
    df=pd.DataFrame(rows).set_index('wid')
    df.to_pickle('drift_feats.pkl')
    print(f"特征表 {df.shape} {time.time()-t0:.0f}s -> drift_feats.pkl", flush=True)

y = df['target']
XY = df[['x_mean','y_mean']]
SPATIAL_FEATS = ['sp_minus_sel','nn_min','nn_med']
ALL = [c for c in df.columns if c not in ('target','x_mean','y_mean')]
NOSP = [c for c in ALL if c not in SPATIAL_FEATS]

def run_variant(cols, splitter_label, splits):
    X = df[cols].fillna(df[cols].median())
    Xs = (X - X.mean()) / (X.std() + 1e-9)
    oof = np.zeros(len(X))
    for tr, va in splits:
        m = Ridge(alpha=10.0); m.fit(Xs.iloc[tr], y.iloc[tr]); oof[va] = m.predict(Xs.iloc[va])
    corr = np.corrcoef(oof, y)[0, 1]
    def pooled(shrink):
        errs = [(sc[w]['pred'] - shrink*o) - sc[w]['truth'] for w, o in zip(X.index, oof)]
        return float(np.sqrt(np.mean(np.concatenate(errs)**2)))
    base = pooled(0.0)
    vals = {s: pooled(s) for s in (0.4, 0.6, 0.8)}
    sbest = min(vals, key=vals.get)
    return corr, base, vals[sbest], sbest

kf5 = list(KFold(5, shuffle=True, random_state=0).split(df))
print("\n=== 消融(KFold over wells) ===")
for name, cols in [("全特征", ALL), ("无空间特征", NOSP)]:
    corr, base, best, sb = run_variant(cols, 'kf', kf5)
    print(f"{name:8s}: OOF corr={corr:.3f}  pooled {base:.4f} -> {best:.4f} (Δ{best-base:+.4f}, shrink={sb})")

print("\n=== 空间分块CV(KMeans 8块按井位, 模拟位置偏移) ===")
km = KMeans(n_clusters=8, n_init=10, random_state=0).fit(XY.values)
blocks = km.labels_
sp_splits = []
for b in range(8):
    va = np.where(blocks == b)[0]; tr = np.where(blocks != b)[0]
    if len(va) > 5: sp_splits.append((tr, va))
for name, cols in [("全特征", ALL), ("无空间特征", NOSP)]:
    corr, base, best, sb = run_variant(cols, 'spatial', sp_splits)
    print(f"{name:8s}: OOF corr={corr:.3f}  pooled {base:.4f} -> {best:.4f} (Δ{best-base:+.4f}, shrink={sb})")

print("\n=== 分半泛化(4种子, 半数井调shrink、另半验证; 无空间特征版) ===")
import random
X = df[NOSP].fillna(df[NOSP].median()); Xs = (X-X.mean())/(X.std()+1e-9)
wells = list(df.index)
for seed in range(4):
    random.seed(seed); half = set(random.sample(wells, len(wells)//2))
    tr_idx = [i for i,w in enumerate(wells) if w in half]
    va_idx = [i for i,w in enumerate(wells) if w not in half]
    m = Ridge(alpha=10.0); m.fit(Xs.iloc[tr_idx], y.iloc[tr_idx])
    pred_va = m.predict(Xs.iloc[va_idx])
    def pooled_va(shrink):
        errs=[(sc[wells[i]]['pred']-shrink*p)-sc[wells[i]]['truth'] for i,p in zip(va_idx,pred_va)]
        return float(np.sqrt(np.mean(np.concatenate(errs)**2)))
    b=pooled_va(0.0); v6=pooled_va(0.6)
    print(f"  seed{seed}: 验证半 base={b:.4f} 修正后={v6:.4f} (Δ{v6-b:+.4f})")
