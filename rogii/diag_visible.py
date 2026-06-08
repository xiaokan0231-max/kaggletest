"""
可见井诊断: 这3口测试井(训练集里有真值)上, 比较
  - 物理模型 tvt_from_contacts (泄漏)   vs 真值
  - selector 无泄漏启发式            vs 真值
对照公榜 7.904, 判断公榜是泄漏驱动还是真实预测驱动。
"""
import heur, numpy as np, glob, os, pandas as pd

DATA = heur.DATA
test_ids = sorted(os.path.basename(f).split('__')[0]
                  for f in glob.glob(os.path.join(DATA, 'test', '*__horizontal_well.csv')))

print(f"可见测试井: {test_ids}\n")
all_phys_err, all_sel_err, all_const_err = [], [], []
for wid in test_ids:
    hw_te, tw_te = heur.load_well(wid, 'test')   # 测试版(TVT隐藏)
    hw_tr, tw_tr = heur.load_well(wid, 'train')  # 训练版(有真值)
    ps = int(hw_te['TVT_input'].notna().sum())
    truth = hw_tr['TVT'].values[ps:].astype(float)
    # 物理模型(泄漏): 需要 train 的 EGFDU + TVT
    try:
        phys = np.asarray(heur.tvt_from_contacts(hw_tr, tw_tr))[ps:].astype(float)
        rmse_phys = np.sqrt(np.mean((phys - truth)**2))
    except Exception as e:
        rmse_phys = float('nan'); phys = np.full_like(truth, np.nan); print(f"  phys err {wid}: {e}")
    # selector 无泄漏
    sel = heur.selector_predict(hw_te, tw_te, n_seeds=64)[ps:].astype(float)
    rmse_sel = np.sqrt(np.mean((sel - truth)**2))
    const = float(hw_te['TVT_input'].dropna().iloc[-1])
    rmse_const = np.sqrt(np.mean((const - truth)**2))
    all_phys_err.append(phys - truth); all_sel_err.append(sel - truth)
    all_const_err.append(np.full_like(truth, const) - truth)
    print(f"{wid}: n_eval={len(truth)}  phys={rmse_phys:6.3f}  selector={rmse_sel:6.3f}  const={rmse_const:6.3f}")

pe = np.concatenate(all_phys_err); se = np.concatenate(all_sel_err); ce = np.concatenate(all_const_err)
print(f"\n3井 pooled RMSE:  phys={np.sqrt((pe**2).mean()):.4f}  "
      f"selector={np.sqrt((se**2).mean()):.4f}  const={np.sqrt((ce**2).mean()):.4f}")
print(f"\n对照: 公榜LB=7.904 (提交=0.3*GBDT + 0.7*tvt2; 可见井 tvt2=物理模型)")
print("若 phys<<selector 且接近LB -> 公榜泄漏驱动; 若 selector 接近LB -> 真实预测驱动")
