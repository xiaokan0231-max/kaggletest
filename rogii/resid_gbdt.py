"""
#6 残差重构试验: 训GBDT预测 selector残差(truth - sel), 看 sel+shrink*residual 能否打败当前blend。
关键判断: selector的错可不可学(残差OOF与真残差的相关性)。
"""
import pandas as pd, numpy as np, pickle, time
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

t0 = time.time()
df = pd.read_parquet('artifacts/data/train.parquet')
print(f"parquet全载 {time.time()-t0:.1f}s, shape={df.shape}", flush=True)

# 对齐 selector
sc = {d['wid']: d for d in pickle.load(open('selcache.pkl', 'rb'))}
sel = np.full(len(df), np.nan, np.float32)
for well, g in df.groupby('well', sort=False):
    if well in sc and len(sc[well]['pred']) == len(g):
        sel[g.index.values] = sc[well]['pred']
mask = ~np.isnan(sel)
df = df[mask].reset_index(drop=True); sel = sel[mask]
print(f"对齐selector后: {len(df)}行 ({df['well'].nunique()}井)", flush=True)

base = df['last_known_tvt'].values.astype(np.float64)
y = df['target'].values.astype(np.float64)          # = truth - last_known
truth = base + y
resid_target = truth - sel                            # selector残差(要学的)
g = df['well'].values
features = [c for c in df.columns if c not in {'well', 'id', 'target'}]
X = df[features]
print(f"特征数={len(features)}  残差target std={resid_target.std():.3f}", flush=True)

# GroupKFold 训残差GBDT
gkf = GroupKFold(n_splits=5)
resid_oof = np.zeros(len(df))
params = dict(n_estimators=400, learning_rate=0.05, num_leaves=128, min_child_samples=40,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
             reg_alpha=0.1, objective='regression', n_jobs=-1, verbose=-1)
for fi, (tr, va) in enumerate(gkf.split(X, resid_target, groups=g)):
    m = lgb.LGBMRegressor(**params)
    m.fit(X.iloc[tr], resid_target[tr])
    resid_oof[va] = m.predict(X.iloc[va])
    print(f"  fold{fi} done {time.time()-t0:.0f}s", flush=True)

# 残差可学性
cr = np.corrcoef(resid_oof, resid_target)[0, 1]
print(f"\ncorr(residual_oof, 真残差) = {cr:.3f}  (>0.3才算可学)", flush=True)

def rmse(p): return float(np.sqrt(np.mean((p - truth)**2)))
print(f"\nselector单独        = {rmse(sel):.4f}")
# 当前blend(用train_side的gbdt_oof)
ts = pickle.load(open('train_side.pkl', 'rb'))
gdf = pd.DataFrame(dict(well=ts['well'], gbdt=ts['gbdt_oof']))
gb = np.full(len(df), np.nan)
# 对齐gbdt_oof(按井顺序)
gi = 0
gmap = {}
for well, gg in gdf.groupby('well', sort=False):
    gmap[well] = gg['gbdt'].values
for well, gg in df.groupby('well', sort=False):
    if well in gmap and len(gmap[well]) == len(gg):
        gb[gg.index.values] = gmap[well]
gmask = ~np.isnan(gb)
print(f"  (gbdt对齐 {gmask.sum()}行)")
print(f"当前blend 0.3gbdt+0.7sel = {np.sqrt(np.mean((0.3*gb[gmask]+0.7*sel[gmask]-truth[gmask])**2)):.4f}")
print(f"全局blend 0.5/0.5        = {np.sqrt(np.mean((0.5*gb[gmask]+0.5*sel[gmask]-truth[gmask])**2)):.4f}")
print("\n--- #6 残差: sel + shrink*residual_oof ---")
best=(1e9,None)
for sh in [0.1,0.2,0.3,0.4,0.5,0.7,1.0]:
    r=rmse(sel+sh*resid_oof)
    if r<best[0]: best=(r,sh)
    print(f"  shrink={sh:.1f}: {r:.4f}")
print(f"残差最优 shrink={best[1]} -> {best[0]:.4f}")
print(f"\n对比: selector={rmse(sel):.3f}  当前blend≈9.59  残差最优={best[0]:.4f}")
np.save('resid_oof.npy', resid_oof)
