# rogii wellbore geology prediction — submission kernel (Code competition)
# 在 Kaggle 上运行: 读 /kaggle/input 隐藏测试集, 输出 /kaggle/working/submission.csv
import os, glob
import numpy as np
import pandas as pd

# 自动发现数据目录: 在 /kaggle/input 下递归找 sample_submission.csv
def _find_base():
    roots = ["/kaggle/input",
             os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "extracted")]
    for root in roots:
        if not os.path.isdir(root):
            continue
        hits = glob.glob(os.path.join(root, "**", "sample_submission.csv"), recursive=True)
        if hits:
            return os.path.dirname(hits[0])
    # 调试: 打印 /kaggle/input 结构
    for root in roots:
        if os.path.isdir(root):
            print("DIR", root, "->", os.listdir(root))
            for d, subs, files in os.walk(root):
                print("  ", d, files[:6])
    raise FileNotFoundError("sample_submission.csv not found under /kaggle/input")

BASE = _find_base()
print("BASE =", BASE)
SAMPLE = os.path.join(BASE, "sample_submission.csv")
# test 文件可能直接在 BASE 下, 也可能在 BASE/test 下
def _find_test_dir():
    for cand in [os.path.join(BASE, "test"), BASE]:
        if glob.glob(os.path.join(cand, "*__horizontal_well.csv")):
            return cand
    hits = glob.glob(os.path.join(BASE, "**", "*__horizontal_well.csv"), recursive=True)
    return os.path.dirname(hits[0]) if hits else os.path.join(BASE, "test")
TEST = _find_test_dir()
print("TEST =", TEST)


def predict_well(h, t):
    """constant carry-forward: PS 之后全部 = 最后一个已知 TVT。
    h: horizontal_well df (MD,X,Y,Z,GR,TVT_input); t: typewell df (TVT,GR).
    返回与 h 等长的预测数组(仅 NaN 区会被取用)。"""
    ti = h["TVT_input"].values.astype(float)
    known = ti[~np.isnan(ti)]
    last = known[-1] if len(known) else 0.0
    pred = np.where(np.isnan(ti), last, ti)
    return pred


def main():
    samp = pd.read_csv(SAMPLE)
    # id = "{well}_{0-based row idx}"
    well = samp["id"].str.rsplit("_", n=1).str[0]
    idx = samp["id"].str.rsplit("_", n=1).str[1].astype(int)
    out = {}
    for wid in sorted(well.unique()):
        h = pd.read_csv(os.path.join(TEST, f"{wid}__horizontal_well.csv"))
        # typewell 文件名可能带后缀, 用 glob 兜底
        tw = glob.glob(os.path.join(TEST, f"{wid}__typewell*.csv"))
        t = pd.read_csv(tw[0]) if tw else None
        pred = predict_well(h, t)
        rows = idx[well == wid].values
        for r in rows:
            out[f"{wid}_{r}"] = pred[r]
    sub = samp[["id"]].copy()
    sub["tvt"] = sub["id"].map(out)
    assert sub["tvt"].notna().all(), "missing predictions"
    sub.to_csv("submission.csv", index=False)
    print("wrote submission.csv", sub.shape)
    print(sub.head())


if __name__ == "__main__":
    main()
