# Iris Flower Classification: Decision Tree + Streamlit

Predicts iris species from sepal/petal measurements with a tuned scikit-learn Decision Tree.
The original notebook (Keras neural network + EDA) is kept in `notebooks/`.

## Structure
`train_model.py` train/evaluate/save · `predict.py` reload & predict · `app.py` Streamlit app ·
`model.joblib` (pipeline) · `metrics.json` · `graphs/` · `data/Iris.csv` · `requirements.txt`

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train_model.py     # optional: regenerates model.joblib, metrics.json, graphs/
python predict.py         # reload test
streamlit run app.py
```

## Method
- Dataset: 150 rows, 4 numeric features, 3 balanced classes, no missing values. `Id` is dropped (an index, not a feature); duplicates removed (149 rows remain).
- Stratified 70/30 split, `random_state=42`. Pipeline = median imputer + DecisionTreeClassifier.
- Hyperparameters tuned with 5-fold stratified CV on the **training set only** (no test leakage).
- Loss = multi-class **log loss** (cross-entropy). Accuracy is reported separately and is not a loss.

## Results (measured, test set n=45)
| Metric | Train | Test |
|---|---|---|
| Accuracy | 97.12% | 88.89% |
| Log loss | 0.0586 | 1.7010 |

Test weighted precision/recall/F1: 0.890 / 0.889 / 0.889. CV accuracy on training data: 96.19%.
Best params: gini, max_depth=3, min_samples_leaf=2. All 5 test errors are versicolor/virginica confusions.

**Interpretation:** the gap between train and test is expected with 149 rows and a 45-sample test set (one sample ≈ 2.2% accuracy), so the test number is noisy. The high test log loss comes from tree leaves giving probability 0 to other classes: a confident wrong prediction is penalized heavily. The notebook's neural network reported ~97.8% on its own split; the numbers are not directly comparable (different model, split, data copy).

## Dataset note
The Kaggle `Iris.csv` was not included in the upload. `data/Iris.csv` was generated from scikit-learn's bundled Iris data in the Kaggle column format. It can differ slightly from the Kaggle copy (a couple of known sample differences), which is why dedup yields 149 rows here versus 147 in the notebook. Replace it with your Kaggle file and re-run `train_model.py` if you want an exact match.

## Deploy on Streamlit Community Cloud (https://share.streamlit.io/)
1. Create a GitHub repo and upload the project contents (repo root must contain `app.py` and `requirements.txt`):
   ```bash
   git init && git add . && git commit -m "Iris decision tree app"
   git branch -M main && git remote add origin https://github.com/<you>/<repo>.git && git push -u origin main
   ```
2. Sign in at https://share.streamlit.io/ with GitHub.
3. Click **Create app**, choose the repo, branch `main`, main file `app.py`, then **Deploy**.
4. Keep `model.joblib` committed. If the Cloud Python/scikit-learn version differs much from the one used to train, pin the scikit-learn version in `requirements.txt` to the one you trained with, or re-train in `requirements.txt`.

## Limitations / unverified
- Streamlit was not installable in the build environment, so the app was **not run**; it was only syntax-checked and its metric keys and file paths were cross-checked. Please run `streamlit run app.py` once locally.
- Not deployed publicly.
- Small dataset; see interpretation above.
