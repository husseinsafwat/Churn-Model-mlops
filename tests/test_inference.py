"""
Smoke test for inference (no pytest). Run from repo root:

    python tests/test_inference.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from inference import FEATURE_COLUMNS, inference  # noqa: E402


def _toy_model():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "CreditScore": rng.integers(400, 800, n),
            "Geography": rng.choice(["France", "Spain", "Germany"], n),
            "Gender": rng.choice(["Male", "Female"], n),
            "Age": rng.integers(20, 70, n),
            "Tenure": rng.integers(0, 10, n),
            "Balance": rng.uniform(0, 100_000, n),
            "NumOfProducts": rng.integers(1, 4, n),
            "HasCrCard": rng.integers(0, 2, n),
            "IsActiveMember": rng.integers(0, 2, n),
            "EstimatedSalary": rng.uniform(20_000, 120_000, n),
        }
    )
    y = ((df["Age"] > 45) & (df["Balance"] > 50_000)).astype(int)
    cat = ["Geography", "Gender"]
    num = [c for c in FEATURE_COLUMNS if c not in cat]
    pre = make_column_transformer(
        (StandardScaler(), num),
        (OneHotEncoder(handle_unknown="ignore", drop="first"), cat),
        remainder="passthrough",
    )
    X = df[FEATURE_COLUMNS]
    Xt = pre.fit_transform(X)
    Xm = pd.DataFrame(Xt, columns=pre.get_feature_names_out())
    clf = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=0)
    clf.fit(Xm, y)
    return pre, clf


def main() -> None:
    pre, clf = _toy_model()
    mv = SimpleNamespace(run_id="x", version="1")

    def load(uri: str):
        return pre if uri.startswith("runs:") else clf

    fake = MagicMock()
    fake.get_latest_versions.return_value = [mv]

    rows = [
        {
            "CreditScore": 700,
            "Geography": "France",
            "Gender": "Male",
            "Age": 50,
            "Tenure": 5,
            "Balance": 90000.0,
            "NumOfProducts": 2,
            "HasCrCard": 1,
            "IsActiveMember": 0,
            "EstimatedSalary": 80000.0,
        },
        {
            "CreditScore": 600,
            "Geography": "Spain",
            "Gender": "Female",
            "Age": 28,
            "Tenure": 2,
            "Balance": 10000.0,
            "NumOfProducts": 1,
            "HasCrCard": 0,
            "IsActiveMember": 1,
            "EstimatedSalary": 40000.0,
        },
    ]

    with patch("inference.mlflow.set_tracking_uri"), patch(
        "inference.MlflowClient", return_value=fake
    ), patch("mlflow.sklearn.load_model", side_effect=load):
        out = inference(rows, "FakeModel", tracking_uri="http://local")

    print(out)
    assert len(out) == 2
    for r in out:
        assert r["prediction"] in (0, 1)
        assert 0.0 <= r["probability"] <= 1.0
    print("ok")


if __name__ == "__main__":
    main()
