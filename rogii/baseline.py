"""
rogii wellbore geology prediction — baseline & local eval.

任务: 预测水平井 PS(Prediction Start) 之后每个测点的 TVT。
- 已知特征(train/test 都有): MD, X, Y, Z, GR, TVT_input(仅 PS 前非空)
- 参考井 typewell: TVT, GR(标准 GR-vs-TVT 曲线)
- 指标: pooled RMSE over all predicted points (dTVT = trueTVT - predTVT)

提交格式: id = "{well}_{0-based行号}", 列 tvt, 仅包含 PS..末尾的行。
"""
import pandas as pd, numpy as np, glob, os

DATA = os.path.join(os.path.dirname(__file__), "data", "extracted")
TRAIN = os.path.join(DATA, "train")
TEST = os.path.join(DATA, "test")


def well_ids(folder):
    return sorted(os.path.basename(f).split("__")[0]
                  for f in glob.glob(os.path.join(folder, "*__horizontal_well.csv")))


def load_well(folder, wid):
    h = pd.read_csv(os.path.join(folder, f"{wid}__horizontal_well.csv"))
    t = pd.read_csv(os.path.join(folder, f"{wid}__typewell.csv"))
    return h, t


def ps_of(h):
    """PS = 已知 TVT_input 的点数; 预测区 = 行 [ps:]"""
    return int(h["TVT_input"].notna().sum())


def predict_constant(h):
    """常数延拓: PS 之后全部 = PS 处最后一个已知 TVT。返回预测区的 ndarray。"""
    ps = ps_of(h)
    known = h["TVT_input"].values[:ps]
    last = known[-1]
    n_pred = len(h) - ps
    return np.full(n_pred, last, dtype=float), ps


def build_submission(predict_fn, out_path):
    rows = []
    for wid in well_ids(TEST):
        h, t = load_well(TEST, wid)
        pred, ps = predict_fn(h, t)
        for j, val in enumerate(pred):
            rows.append((f"{wid}_{ps + j}", val))
    sub = pd.DataFrame(rows, columns=["id", "tvt"])
    # 对齐官方 sample_submission 顺序
    samp = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
    sub = samp[["id"]].merge(sub, on="id", how="left")
    assert sub["tvt"].notna().all(), "有缺失预测!"
    sub.to_csv(out_path, index=False)
    print(f"写出 {out_path}: {len(sub)} 行")
    return sub


def eval_on_test_wells(predict_fn):
    """用 train 里同 id 的真值评估测试井(泄露真值, 仅本地用)。返回 pooled RMSE。"""
    errs = []
    per = {}
    for wid in well_ids(TEST):
        h_te, t_te = load_well(TEST, wid)
        pred, ps = predict_fn(h_te, t_te)
        h_tr, _ = load_well(TRAIN, wid)      # 含完整 TVT
        truth = h_tr["TVT"].values[ps:]
        e = pred - truth
        errs.append(e)
        per[wid] = float(np.sqrt(np.mean(e ** 2)))
    pooled = float(np.sqrt(np.mean(np.concatenate(errs) ** 2)))
    return pooled, per


if __name__ == "__main__":
    pf = lambda h, t: predict_constant(h)
    pooled, per = eval_on_test_wells(pf)
    print("=== 常数延拓 在 3 口测试井上的真实 RMSE ===")
    for w, r in per.items():
        print(f"  {w}: {r:.3f}")
    print(f"  pooled RMSE = {pooled:.3f}")
    build_submission(pf, os.path.join(os.path.dirname(__file__), "sub_constant.csv"))
