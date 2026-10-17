"""Train, evaluate and save a Decision Tree classifier for the Iris dataset."""
import json
from pathlib import Path
import joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             log_loss, precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree

SEED = 42
ROOT = Path(__file__).parent
DATA, GRAPHS = ROOT / "data" / "Iris.csv", ROOT / "graphs"
FEATURES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
TARGET = "Species"
GRAPHS.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")

def save(fig, name):
    fig.tight_layout(); fig.savefig(GRAPHS / name, dpi=150); plt.close(fig)

def make_pipeline(**params):
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("clf", DecisionTreeClassifier(random_state=SEED, **params))])

def main():
    df = pd.read_csv(DATA)
    raw_rows = len(df)
    df = df.drop(columns="Id").drop_duplicates().reset_index(drop=True)  # Id is an index, not a feature
    print(f"Rows: {raw_rows} -> {len(df)} after dropping duplicates; missing={int(df.isna().sum().sum())}")
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)

    # Hyperparameter tuning: cross-validation on the TRAINING set only (no test-set leakage)
    grid = {"clf__criterion": ["gini", "entropy"], "clf__max_depth": [2, 3, 4, 5, 6, None],
            "clf__min_samples_leaf": [1, 2, 3, 5, 8], "clf__min_samples_split": [2, 4, 8],
            "clf__ccp_alpha": [0.0, 0.01, 0.02]}
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    gs = GridSearchCV(make_pipeline(), grid, cv=cv, scoring="accuracy", n_jobs=-1).fit(X_tr, y_tr)
    model = gs.best_estimator_
    best = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
    print("Best params:", best, "| CV accuracy: %.4f" % gs.best_score_)

    classes = list(model.classes_)
    p_tr, p_te = model.predict_proba(X_tr), model.predict_proba(X_te)
    yp_tr, yp_te = model.predict(X_tr), model.predict(X_te)
    prec, rec, f1, _ = precision_recall_fscore_support(y_te, yp_te, average="weighted", zero_division=0)
    cm = confusion_matrix(y_te, yp_te, labels=classes)
    report = classification_report(y_te, yp_te, output_dict=True, zero_division=0)
    metrics = {
        "train_accuracy": accuracy_score(y_tr, yp_tr), "test_accuracy": accuracy_score(y_te, yp_te),
        "train_log_loss": log_loss(y_tr, p_tr, labels=classes), "test_log_loss": log_loss(y_te, p_te, labels=classes),
        "test_precision_weighted": prec, "test_recall_weighted": rec, "test_f1_weighted": f1,
        "cv_accuracy_mean_on_train": gs.best_score_, "confusion_matrix": cm.tolist(), "classes": classes,
        "classification_report": report, "best_params": best,
        "n_rows_raw": raw_rows, "n_rows_after_dedup": len(df), "n_train": len(X_tr), "n_test": len(X_te),
        "class_counts": y.value_counts().to_dict(), "features": FEATURES, "random_state": SEED,
        "feature_ranges": {c: [float(X[c].min()), float(X[c].max())] for c in FEATURES},
        "feature_importance": dict(zip(FEATURES, map(float, model.named_steps["clf"].feature_importances_))),
        "tree_depth": int(model.named_steps["clf"].get_depth()), "tree_leaves": int(model.named_steps["clf"].get_n_leaves()),
        "loss_definition": "Multi-class log loss (cross-entropy): -(1/N) sum_i log p(true class of i). Lower is better.",
    }
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump({"pipeline": model, "features": FEATURES, "classes": classes}, ROOT / "model.joblib")

    # Loss/accuracy vs. tree depth (other tuned params fixed) - the natural "training curve" for a tree
    depths = list(range(1, 11)); curve = {k: [] for k in ["tr_l", "te_l", "tr_a", "te_a"]}
    fixed = {k: v for k, v in best.items() if k != "max_depth"}
    for d in depths:
        m = make_pipeline(max_depth=d, **fixed).fit(X_tr, y_tr)
        a, b = m.predict_proba(X_tr), m.predict_proba(X_te)
        curve["tr_l"].append(log_loss(y_tr, a, labels=classes)); curve["te_l"].append(log_loss(y_te, b, labels=classes))
        curve["tr_a"].append(m.score(X_tr, y_tr)); curve["te_a"].append(m.score(X_te, y_te))
    for key, (t, e), ttl, yl, fn in [("l", ("tr_l", "te_l"), "Training vs Testing Log Loss by Tree Depth", "Log loss (lower is better)", "loss_curve.png"),
                                     ("a", ("tr_a", "te_a"), "Training vs Testing Accuracy by Tree Depth", "Accuracy", "accuracy_curve.png")]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(depths, curve[t], "o-", label="Training"); ax.plot(depths, curve[e], "s--", label="Testing")
        ax.axvline(metrics["tree_depth"], color="gray", ls=":", label=f"Final model depth ({metrics['tree_depth']})")
        ax.set(title=ttl, xlabel="max_depth", ylabel=yl, xticks=depths); ax.legend(); save(fig, fn)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set(title="Confusion Matrix (Test Set)", xlabel="Predicted", ylabel="Actual"); plt.setp(ax.get_xticklabels(), rotation=20); save(fig, "confusion_matrix.png")

    imp = pd.Series(metrics["feature_importance"]).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4)); imp.plot.barh(ax=ax, color="#2a9d8f")
    ax.set(title="Feature Importance (Decision Tree)", xlabel="Importance (Gini/entropy reduction)", ylabel="Feature"); save(fig, "feature_importance.png")

    fig, ax = plt.subplots(figsize=(16, 9))
    plot_tree(model.named_steps["clf"], feature_names=FEATURES, class_names=classes, filled=True, rounded=True, fontsize=9, ax=ax)
    ax.set_title("Decision Tree Structure"); save(fig, "decision_tree.png")

    print(f"Train acc {metrics['train_accuracy']:.4f} | Test acc {metrics['test_accuracy']:.4f}")
    print(f"Train log loss {metrics['train_log_loss']:.4f} | Test log loss {metrics['test_log_loss']:.4f}")
    print(f"Test precision/recall/F1 (weighted): {prec:.4f}/{rec:.4f}/{f1:.4f}")
    print(classification_report(y_te, yp_te, zero_division=0)); print("Confusion matrix:\n", cm)

if __name__ == "__main__":
    main()
