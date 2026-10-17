"""Reload the saved model and predict on new input."""
from pathlib import Path
import joblib, pandas as pd

def load(path=Path(__file__).parent / "model.joblib"):
    return joblib.load(path)

def predict(bundle, rows):
    X = pd.DataFrame(rows, columns=bundle["features"])
    pipe = bundle["pipeline"]
    return pipe.predict(X), pipe.predict_proba(X)

if __name__ == "__main__":
    b = load()
    labels, proba = predict(b, [[5.1, 3.5, 1.4, 0.2], [6.0, 2.8, 4.5, 1.4], [6.9, 3.1, 5.8, 2.1]])
    for l, p in zip(labels, proba):
        print(l, dict(zip(b["classes"], p.round(3))))
