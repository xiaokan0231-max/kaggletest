"""
无泄漏训练井上求 GBDT分支 vs selector分支 的最优融合权重。
train_side.pkl: GBDT分支OOF预测; selcache.pkl: selector预测。
公开方案用 0.3*GBDT+0.7*sel(为可见井物理模型调的), 这里看无泄漏井上最优是多少。
"""
import pickle, numpy as np, pandas as pd

ts = pickle.load(open('train_side.pkl', 'rb'))
sel_cache = {d['wid']: d for d in pickle.load(open('selcache.pkl', 'rb'))}

df = pd.DataFrame(dict(well=ts['well'], base=ts['base'], ytrue=ts['ytrue'],
                       gbdt=ts['gbdt_oof'], md_since=ts['md_since']))

# 按井对齐 selector 预测
sel_col = np.full(len(df), np.nan)
matched_wells = 0
i = 0
for well, g in df.groupby('well', sort=False):
    idx = g.index.values
    if well in sel_cache:
        sp = sel_cache[well]['pred']
        if len(sp) == len(idx):
            sel_col[idx] = sp
            matched_wells += 1
df['sel'] = sel_col

print(f"总行={len(df)}  对齐selector的井={matched_wells}  有sel的行={df['sel'].notna().sum()}")
# 只在两者都有的行上比较
m = df['sel'].notna().values
g = df['gbdt'].values[m]; s = df['sel'].values[m]; y = df['ytrue'].values[m]
def rmse(p): return float(np.sqrt(np.mean((p - y) ** 2)))
print(f"\n在 {m.sum()} 行(无泄漏)上:")
print(f"  GBDT分支单独   = {rmse(g):.4f}")
print(f"  selector单独   = {rmse(s):.4f}")
print(f"  公开方案0.3/0.7 = {rmse(0.3*g + 0.7*s):.4f}")
print("\n--- 扫描融合权重 w*gbdt+(1-w)*sel ---")
best = (1e9, None)
for w in np.arange(0.0, 1.01, 0.05):
    r = rmse(w*g + (1-w)*s)
    if r < best[0]: best = (r, w)
    mark = "  <== 当前0.3" if abs(w-0.3) < 1e-6 else ("  <== 最优" if False else "")
    print(f"  w_gbdt={w:.2f}: {r:.4f}{mark}")
print(f"\n最优: w_gbdt={best[1]:.2f} -> {best[0]:.4f}  (vs 0.3/0.7的 {rmse(0.3*g+0.7*s):.4f})")
