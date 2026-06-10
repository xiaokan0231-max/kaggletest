"""
井级净漂移回归可行性诊断。
发现: selector误差59%是每井漂移均值(oracle-C: 10.51->6.72)。
问题: 漂移均值能否用"测试时可得"的井级特征预测? 哪怕弱预测(R2~0.1), shrink修正后也稳赚。
target = mean(sel_pred - truth) per well; 特征全部无truth依赖。
验证: KFold over wells + shrink扫描 + 分半稳健性。
"""
import numpy as np, pickle, pandas as pd, time
from joblib import Parallel, delayed
import heur

sc = {d['wid']: d for d in pickle.load(open('selcache.pkl', 'rb'))}
dg = {r['wid']: r for r in pickle.load(open('distgate.pkl', 'rb'))}
try:
    c2 = {d['wid']: d for d in pickle.load(open('selcache2.pkl', 'rb'))}
except Exception:
    c2 = {}

def feats(wid):
    if wid not in sc:
        return None
    try:
        hw, tw = heur.load_well(wid, 'train')
    except Exception:
        return None
    d = sc[wid]
    target = float(np.mean(d['pred'] - d['truth']))          # 净漂移(要预测的)
    ps = int(hw['TVT_input'].notna().sum())
    kn = hw.iloc[:ps]; ev = hw.iloc[ps:]
    md = hw['MD'].values; z = hw['Z'].values
    twg = tw['GR'].dropna().values; twt = tw.sort_values('TVT')
    tw_tvt = twt['TVT'].values; tw_gr = twt['GR'].fillna(twt['GR'].mean()).values
    # 前缀尾段动态
    tail = kn.tail(40)
    sl = np.polyfit(tail['MD'], tail['TVT_input'], 1)[0] if len(tail) > 5 else 0.0
    tail2 = kn.tail(120)
    sl2 = np.polyfit(tail2['MD'], tail2['TVT_input'], 1)[0] if len(tail2) > 10 else 0.0
    # selector预测自身的形状特征(无truth)
    p = d['pred']
    pred_slope = np.polyfit(np.arange(len(p)), p, 1)[0] if len(p) > 10 else 0.0
    pred_net = float(p[-1] - p[0])
    # GR错位: 前缀GR vs typewell在已知TVT处的GR
    gr_at_k = np.interp(kn['TVT_input'].values, tw_tvt, tw_gr)
    gr_mis = float(np.nanmean(kn['GR'].values - gr_at_k))
    f = dict(wid=wid, target=target,
             tail_slope=sl, tail_slope_long=sl2, slope_diff=sl - sl2,
             pred_slope=pred_slope, pred_net=pred_net,
             ps_frac=ps / len(hw), n_ev=len(ev), md_span_ev=md[-1] - md[ps - 1],
             z_net_ev=float(z[-1] - z[ps - 1]), z_span_ev=float(np.ptp(z[ps:])),
             gr_mis=gr_mis, gr_ev_m=float(np.nanmean(ev['GR'])), gr_kn_m=float(np.nanmean(kn['GR'])),
             gr_ev_minus_kn=float(np.nanmean(ev['GR']) - np.nanmean(kn['GR'])),
             twg_m=float(twg.mean()), twg_rng=float(np.ptp(twg)),
             tvt_last=float(kn['TVT_input'].iloc[-1]),
             tvt_pos_in_tw=float((kn['TVT_input'].iloc[-1] - tw_tvt.min()) / max(np.ptp(tw_tvt), 1e-9)))
    # spatial信号(distgate): spatial与selector的预测分歧均值 = mean(sp_err)-mean(sel_err)+target... 不对,
    # mean(sp_pred-sel_pred) = mean(sp_err - sel_err), 无truth依赖? sp_err含truth但差里truth消掉 -> OK
    if wid in dg:
        r = dg[wid]
        f['sp_minus_sel'] = float(np.mean(r['sp_err']) - np.mean(np.asarray(d['pred'] - d['truth'])))  # = mean(sp_pred-sel_pred)
        f['nn_min'] = r['nn_min']; f['nn_med'] = r['nn_med']
    else:
        f['sp_minus_sel'] = 0.0; f['nn_min'] = 9999.0; f['nn_med'] = 9999.0
    # PF集成分歧(cache2)
    if wid in c2:
        f['disagree'] = float(np.mean(c2[wid]['disagree']))
        f['beam_minus_sel'] = float(np.mean(c2[wid]['beam'] - c2[wid]['pred']))
    else:
        f['disagree'] = np.nan; f['beam_minus_sel'] = np.nan
    return f

t0 = time.time()
rows = [r for r in Parallel(n_jobs=-1)(delayed(feats)(w) for w in sc.keys()) if r]
df = pd.DataFrame(rows).set_index('wid')
print(f"特征表 {df.shape} {time.time()-t0:.0f}s", flush=True)
y = df.pop('target')
X = df.fillna(df.median())

# 单特征相关性
print("\n单特征 vs 净漂移 corr:")
cors = X.apply(lambda c: np.corrcoef(c, y)[0, 1]).sort_values(key=abs, ascending=False)
print(cors.head(10).round(3).to_string())

# KFold 回归(LightGBM + Ridge对照)
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import lightgbm as lgb
kf = KFold(n_splits=5, shuffle=True, random_state=0)
oof_l = np.zeros(len(X)); oof_r = np.zeros(len(X))
for tr, va in kf.split(X):
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15,
                          min_child_samples=30, subsample=0.8, colsample_bytree=0.7,
                          reg_lambda=5.0, verbose=-1)
    m.fit(X.iloc[tr], y.iloc[tr]); oof_l[va] = m.predict(X.iloc[va])
    r = Ridge(alpha=10.0);
    Xs = (X - X.mean()) / (X.std() + 1e-9)
    r.fit(Xs.iloc[tr], y.iloc[tr]); oof_r[va] = r.predict(Xs.iloc[va])
print(f"\nOOF corr(预测漂移, 真漂移): LGBM={np.corrcoef(oof_l,y)[0,1]:.3f}  Ridge={np.corrcoef(oof_r,y)[0,1]:.3f}")

# shrink修正后的pooled RMSE
def pooled_after(oof, shrink):
    errs = []
    for wid, drift_hat in zip(X.index, oof):
        d = sc[wid]
        errs.append((d['pred'] - shrink * drift_hat) - d['truth'])
    e = np.concatenate(errs)
    return float(np.sqrt(np.mean(e**2)))
base = pooled_after(np.zeros(len(X)), 0)
print(f"\nbaseline pooled = {base:.4f}")
best=(base,0,None)
for name, oof in [('LGBM', oof_l), ('Ridge', oof_r)]:
    for s in [0.2, 0.4, 0.6, 0.8, 1.0]:
        v = pooled_after(oof, s)
        tag = " <==" if v < best[0] else ""
        if v < best[0]: best=(v,s,name)
        print(f"  {name} shrink={s}: {v:.4f} (Δ{v-base:+.4f}){tag}")
print(f"\nBEST: {best[2]} shrink={best[1]} -> {best[0]:.4f}  (oracle上限=6.72)")
