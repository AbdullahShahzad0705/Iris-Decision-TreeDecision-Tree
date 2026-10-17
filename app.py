"""Streamlit app: Iris Decision Tree classifier. Loads saved artifacts; never retrains."""
import json
from pathlib import Path
import joblib, pandas as pd, streamlit as st

ROOT = Path(__file__).parent
MODEL_PATH, METRICS_PATH, DATA_PATH, GRAPHS = (ROOT / "model.joblib", ROOT / "metrics.json",
                                              ROOT / "data" / "Iris.csv", ROOT / "graphs")
st.set_page_config(page_title="Iris Decision Tree", page_icon="🌸", layout="wide")
st.markdown("""<style>
.block-container{padding-top:2rem;max-width:1150px}
div[data-testid="stMetric"]{background:#f4f8f7;border:1px solid #d6e6e2;border-radius:12px;padding:14px 16px}
div[data-testid="stMetric"] *{color:#1d3b36 !important}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics():
    return json.loads(METRICS_PATH.read_text())

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

missing = [p.name for p in (MODEL_PATH, METRICS_PATH) if not p.exists()]
if missing:
    st.error(f"Missing file(s): {', '.join(missing)}. Run `python train_model.py` first.")
    st.stop()
bundle, m = load_model(), load_metrics()
features, classes, ranges = bundle["features"], bundle["classes"], m["feature_ranges"]

def graph(name, caption):
    p = GRAPHS / name
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)
    else:
        st.warning(f"{name} not found. Re-run train_model.py.")

st.title("🌸 Iris Flower Classifier")
st.caption("Decision Tree classification with scikit-learn")
tabs = st.tabs(["Overview", "Metrics & Graphs", "Predict", "Limitations"])

with tabs[0]:
    st.subheader("Project overview")
    st.write("This app predicts an iris species (setosa, versicolor, virginica) from four flower measurements "
             "using a tuned Decision Tree. The model and preprocessing are loaded from `model.joblib`.")
    c = st.columns(4)
    c[0].metric("Rows used", m["n_rows_after_dedup"]); c[1].metric("Features", len(features))
    c[2].metric("Train / Test", f"{m['n_train']} / {m['n_test']}"); c[3].metric("Classes", len(classes))
    if DATA_PATH.exists():
        df = load_data()
        st.dataframe(df.head(10), use_container_width=True)
        st.write("**Class counts** (raw file):"); st.bar_chart(df["Species"].value_counts())
        st.dataframe(df.drop(columns="Id").describe().T, use_container_width=True)
    else:
        st.warning("Dataset file not found; dataset preview unavailable.")
    st.write("**Selected hyperparameters** (5-fold CV on training data):", m["best_params"])

with tabs[1]:
    st.subheader("Evaluation")
    a = st.columns(4)
    a[0].metric("Train accuracy", f"{m['train_accuracy']:.2%}"); a[1].metric("Test accuracy", f"{m['test_accuracy']:.2%}")
    a[2].metric("Train log loss", f"{m['train_log_loss']:.3f}"); a[3].metric("Test log loss", f"{m['test_log_loss']:.3f}")
    b = st.columns(4)
    b[0].metric("Precision (wtd)", f"{m['test_precision_weighted']:.3f}"); b[1].metric("Recall (wtd)", f"{m['test_recall_weighted']:.3f}")
    b[2].metric("F1 (wtd)", f"{m['test_f1_weighted']:.3f}"); b[3].metric("CV accuracy (train)", f"{m['cv_accuracy_mean_on_train']:.2%}")
    st.info(f"**Loss definition:** {m['loss_definition']} Accuracy is reported separately and is not a loss. "
            "A tree's leaves output hard class frequencies, so a confident wrong prediction inflates test log loss.")
    c1, c2 = st.columns(2)
    with c1: graph("loss_curve.png", "Log loss vs tree depth")
    with c2: graph("accuracy_curve.png", "Accuracy vs tree depth")
    c3, c4 = st.columns(2)
    with c3: graph("confusion_matrix.png", "Confusion matrix (test set)")
    with c4: graph("feature_importance.png", "Feature importance")
    with st.expander("Decision tree structure"): graph("decision_tree.png", "Fitted tree")
    with st.expander("Classification report (test set)"):
        st.dataframe(pd.DataFrame(m["classification_report"]).T.round(3), use_container_width=True)

with tabs[2]:
    st.subheader("Predict a species")
    cols = st.columns(4); vals = {}
    for col, f in zip(cols, features):
        lo, hi = ranges[f]
        vals[f] = col.number_input(f, value=round((lo + hi) / 2, 1), step=0.1, format="%.1f", help=f"Training range: {lo}–{hi} cm")
    if st.button("Predict", type="primary"):
        bad = [f for f, v in vals.items() if v <= 0]
        if bad:
            st.error(f"Measurements must be positive: {', '.join(bad)}.")
        else:
            outside = [f"{f} ({v} not in {ranges[f][0]}–{ranges[f][1]})" for f, v in vals.items() if not ranges[f][0] <= v <= ranges[f][1]]
            if outside:
                st.warning("Outside the training range, so the prediction is an extrapolation: " + "; ".join(outside))
            X = pd.DataFrame([vals], columns=features)
            pipe = bundle["pipeline"]
            pred, proba = pipe.predict(X)[0], pipe.predict_proba(X)[0]
            st.success(f"Predicted species: **{pred}**")
            st.bar_chart(pd.Series(proba, index=classes, name="Probability"))
            st.dataframe(pd.DataFrame({"Class": classes, "Probability": proba.round(3)}), hide_index=True)

with tabs[3]:
    st.subheader("Limitations")
    st.markdown(f"""
- Tiny dataset ({m['n_rows_after_dedup']} rows; test set only {m['n_test']} samples), so test metrics vary a lot with the split: each test sample is ~{100/m['n_test']:.1f}% accuracy.
- Train accuracy ({m['train_accuracy']:.1%}) exceeds test accuracy ({m['test_accuracy']:.1%}): a mild sign of overfitting.
- Versicolor and virginica overlap; most errors happen between them.
- Decision Tree probabilities are leaf class frequencies, not calibrated confidences.
- Inputs far outside the training ranges are extrapolations. This is an educational model, not for botanical identification.
""")
