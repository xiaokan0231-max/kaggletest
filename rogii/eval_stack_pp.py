from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from koolbox import Trainer
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"


def apply_pp(df, md, pd_, alpha, tau, w_pf):
    d = md * (1 - w_pf) + pd_ * w_pf
    if tau:
        d *= 1.0 - np.exp(-np.maximum(df["md_since"].values, 0.0) / tau)
    return d * alpha


def main():
    print("sklearn", sklearn.__version__, "scipy", scipy.__version__)
    train_df = pd.read_csv(ARTIFACTS / "data" / "train.csv", low_memory=False)
    features = [c for c in train_df.columns if c not in {"well", "id", "target"}]
    y = train_df["target"]
    g = train_df["well"]

    model_names = ["lightgbm-1", "lightgbm-2", "lightgbm-3", "catboost-1", "catboost-2"]
    oof_preds = {}
    for name in model_names:
        pkl = next((ARTIFACTS / "models" / name).glob("*.pkl"))
        trainer = joblib.load(pkl)
        print(f"Loaded {name} with overall RMSE: {trainer.overall_score:.4f}")
        oof_preds[name] = trainer.oof_preds

    ridge_params = {
        "random_state": 42,
        "alpha": 1.6602834637650032,
        "tol": 0.0005030247295617308,
        "positive": True,
        "fit_intercept": True,
    }
    ridge_trainer = Trainer(
        Ridge(**ridge_params),
        task="regression",
        metric=root_mean_squared_error,
        cv=GroupKFold(n_splits=5),
        cv_args={"groups": g},
        verbose=False,
    )
    ridge_trainer.fit(pd.DataFrame(oof_preds), y)
    ridge_oof_preds = ridge_trainer.oof_preds
    print(f"Ridge overall score: {ridge_trainer.overall_score:.5f}")

    pp_params = {"alpha": 1.0, "tau": 85, "w_pf": 0.09}
    base = train_df["last_known_tvt"].values
    ytrue = y.values + base
    pf_oof = train_df["pf_ancc"].values - base

    baseline_d = apply_pp(train_df, ridge_oof_preds, pf_oof, **pp_params)
    baseline_score = root_mean_squared_error(ytrue, base + baseline_d)
    selected_score = baseline_score
    selected_params = pp_params.copy()

    for alpha in [0.98, 0.99, 1.0, 1.01, 1.02]:
        for tau in [35, 50, 65, 85, 105, 130, 170, 220]:
            for w_pf in [0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.16]:
                params = {"alpha": alpha, "tau": tau, "w_pf": w_pf}
                score = root_mean_squared_error(
                    ytrue,
                    base + apply_pp(train_df, ridge_oof_preds, pf_oof, **params),
                )
                if score < selected_score - 1e-9:
                    selected_score = score
                    selected_params = params

    print(f"Baseline pp score: {baseline_score:.5f} params={pp_params}")
    print(f"Selected pp score: {selected_score:.5f} params={selected_params}")


if __name__ == "__main__":
    main()
