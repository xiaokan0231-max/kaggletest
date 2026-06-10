"""
净漂移回归的全诚实空间分块CV(补上鞭刑漏洞)。
漏洞: 之前sp_minus_sel特征用原始LOO插值(邻井含同区井), 分块CV只检验了回归泛化, 没检验特征在位置偏移下的存活。
本版: 留出区井的spatial特征只用区外井插值重算(真·位置偏移模拟)。
同时target升级为最终blend(0.3*gbdt+0.7*sel)的每井漂移——那才是要修的东西。
"""
import numpy as np, pickle, pandas as pd, time
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
import heur, spatial_recon as sr

sc = {d['wid']: d for d in pickle.load(open('selcache.pkl', 'rb'))}
ts = pickle.load(open('train_side.pkl', 'rb'))
feats_df = pd.read_pickle('drift_feats.pkl')   # 已有自井特征(无泄漏), 含x_mean/y_mean

# blend每井漂移 target
gdf = pd.DataFrame(dict(well=ts['well'], ytrue=ts['ytrue'], gbdt=ts['gbdt_oof']))
blend_drift = {}; blend_err = {}
for w, g in gdf.groupby('well', sort=False):
    if w in sc and len(sc[w]['pred']) == len(g):
        be = 0.3*g['gbdt'].values + 0.7*sc[w]['pred'] - g['ytrue'].values
        blend_drift[w] = float(np.mean(be)); blend_err[w] = be.astype(np.float32)
wells = [w for w in feats_df.index if w in blend_drift]
df = feats_df.loc[wells].copy()
df['target_blend'] = [blend_drift[w] for w in wells]
print(f"对齐 {len(df)}井; blend漂移 std={df['target_blend'].std():.2f}", flush=True)
base_pooled = float(np.sqrt(np.mean(np.concatenate([blend_err[w] for w in wells])**2)))
oracle = float(np.sqrt(np.mean(np.concatenate([blend_err[w]-blend_drift[w] for w in wells])**2)))
print(f"blend pooled={base_pooled:.4f}  oracle-C上限={oracle:.4f}", flush=True)

# 空间分块
km = KMeans(n_clusters=8, n_init=10, random_state=0).fit(df[['x_mean','y_mean']].values)
df['block'] = km.labels_

# 每块: 留出区井的spatial特征只用区外井重算
ids_all = heur.well_ids('train')
allX, allS, allW = sr.load_all_surface_samples(ids_all, stride=15)
print(f"面样本 {len(allX)}", flush=True)

def sp_feat_excluding(wid, excl_set, k=12):
    """用'不在excl_set的井'插值EGFDU, 返回 mean(sp_pred - sel_pred)。"""
    try:
        hw, tw = heur.load_well(wid, 'train')
    except Exception:
        return None
    keep = ~np.isin(allW, list(excl_set))
    X2, S2, W2 = allX[keep], allS[keep], allW[keep]
    if len(X2) < 100: return None
    t2 = cKDTree(X2)
    ps = int(hw['TVT_input'].notna().sum())
    Z = hw['Z'].values.astype(float); pts = hw[['X','Y']].values.astype(float)
    tvti = hw['TVT_input'].values.astype(float)
    si = sr.SURF.index('EGFDU')
    nq = min(len(X2), k+50)
    dd, ii = t2.query(pts, k=nq)
    surf = np.empty(len(hw))
    for i in range(len(hw)):
        jj = ii[i][:k]; d2 = dd[i][:k]
        ww = 1.0/(d2+1.0); ww /= ww.sum()
        surf[i] = float((ww*S2[jj, si]).sum())
    raw = -(Z - surf); C = np.nanmean(tvti[:ps] - raw[:ps])
    sp = (raw + C)[ps:]
    sel = sc[wid]['pred']
    if len(sp) != len(sel): return None
    return float(np.mean(sp - sel))

t0 = time.time()
sp_honest = {}
for b in sorted(df['block'].unique()):
    block_wells = set(df.index[df['block'] == b])
    res = Parallel(n_jobs=-1)(delayed(sp_feat_excluding)(w, block_wells) for w in block_wells)
    for w, v in zip(block_wells, res):
        sp_honest[w] = v
    print(f"  block{b}: {len(block_wells)}井 done {time.time()-t0:.0f}s", flush=True)
df['sp_minus_sel_honest'] = [sp_honest.get(w, np.nan) for w in df.index]
ok = df['sp_minus_sel_honest'].notna()
print(f"诚实特征覆盖 {ok.sum()}/{len(df)}", flush=True)
print(f"corr(诚实sp_minus_sel, blend漂移) = {np.corrcoef(df.loc[ok,'sp_minus_sel_honest'], df.loc[ok,'target_blend'])[0,1]:+.3f}")
print(f"corr(原LOO sp_minus_sel, blend漂移) = {np.corrcoef(df.loc[ok,'sp_minus_sel'], df.loc[ok,'target_blend'])[0,1]:+.3f}")

# 全诚实分块CV: 训练用区外井(其特征也用各自LOO即可, 近似), 预测留出区用honest特征
NOSP = [c for c in feats_df.columns if c not in ('target','x_mean','y_mean','sp_minus_sel','nn_min','nn_med')]
COLS = NOSP + ['sp_minus_sel']
d2 = df[ok].copy()
oof = np.zeros(len(d2)); idx_map = {w: i for i, w in enumerate(d2.index)}
for b in sorted(d2['block'].unique()):
    va_w = d2.index[d2['block'] == b]; tr_w = d2.index[d2['block'] != b]
    Xtr = d2.loc[tr_w, COLS].fillna(d2[COLS].median())
    # 验证特征: sp_minus_sel 用 honest 版
    Xva = d2.loc[va_w, COLS].copy()
    Xva['sp_minus_sel'] = d2.loc[va_w, 'sp_minus_sel_honest']
    Xva = Xva.fillna(d2[COLS].median())
    mu, sd = Xtr.mean(), Xtr.std()+1e-9
    m = Ridge(alpha=10.0); m.fit((Xtr-mu)/sd, d2.loc[tr_w,'target_blend'])
    oof[[idx_map[w] for w in va_w]] = m.predict((Xva-mu)/sd)
corr = np.corrcoef(oof, d2['target_blend'])[0,1]
def pooled(shrink):
    errs = [blend_err[w] - shrink*oof[idx_map[w]] for w in d2.index]
    return float(np.sqrt(np.mean(np.concatenate(errs)**2)))
b0 = pooled(0.0)
print(f"\n=== 全诚实空间分块CV(blend target) ===")
print(f"OOF corr={corr:.3f}  base={b0:.4f}")
for s in (0.3, 0.5, 0.7, 0.9):
    v = pooled(s)
    print(f"  shrink={s}: {v:.4f} (Δ{v-b0:+.4f})")
