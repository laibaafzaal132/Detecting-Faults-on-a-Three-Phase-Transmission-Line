# Transmission-Line Fault Detection: Logistic Regression vs SVM

Applied Machine Learning, Assignment 2. From one sample of `Ia, Ib, Ic, Va, Vb, Vc` we decide whether a three-phase
transmission line is faulted. We compare a hand-written logistic regression (six raw features, **Set R**; twelve features
with squares, **Set Q**) against linear and RBF support vector machines.

**Team:** Laiba Afzaal (23F-6073) · Muhammad Ali (23F-6020)
**Blog:** `blog/blog.pdf` / `blog/blog.html` (also `blog/blog.md` for Medium) · **LinkedIn:** `linkedin_post.md`

## Headline results (test set, 2,401 rows, used once)

| Model | Accuracy | TN | False alarms | Missed faults | TP |
|---|---|---|---|---|---|
| Logistic regression, Set R | 72.64 % | 1305 | 0 | 657 | 439 |
| Logistic regression, Set Q | 97.88 % | 1299 | 6 | 45 | 1051 |
| Linear SVM (C=100) | 73.64 % | 1305 | 0 | 633 | 463 |
| **RBF SVM (C=100, γ=1)** | **99.58 %** | 1301 | 4 | **6** | 1090 |

All five tables: [`results/tables.md`](results/tables.md). Task 6 answers: [`results/analysis.md`](results/analysis.md).

## Repository layout

```
data/detect_dataset.csv     the Kaggle dataset
src/load_data.py            the loader (Task 1.1)
src/prepare.py              Task 1: cleaning, 60/20/20 split (seed 42), standardisation
src/eda.py                  Task 2: exploratory plots
src/logistic.py             sigmoid, hypothesis, cost, fit_logistic, predict (by hand)
src/metrics.py              accuracy, confusion_matrix (by hand)
src/train_logistic.py       Task 3: choose alpha, train Set R / Set Q, save theta
src/train_svm.py            Task 4: 4 linear + 8 RBF SVMs, save the best of each
src/test_models.py          Task 5: test set, used once; writes results/tables.md
run_all.py                  runs Tasks 1-5 in order
results/                    tables.md, table*.csv, analysis.md, figures/, models/ (theta + .joblib)
app/                        Task 7 real-time front end (Flask app.py + templates/index.html)
blog/                       blog.html + blog.pdf + static/ figures, blog.md (Medium source)
notebooks/                  the Colab notebook with all outputs
```

## Run it in VS Code

1. Open the folder: **File → Open Folder → `fault-detection`**.
2. Open a terminal (**Ctrl + `**) and create the environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Pick the interpreter: **Ctrl+Shift+P → Python: Select Interpreter → `.venv`**.
4. Reproduce Tasks 1-5 (rebuilds every table, figure and saved model):
   ```bash
   python run_all.py
   ```
   or run the steps one by one: `python src/train_logistic.py`, `python src/train_svm.py`, `python src/test_models.py`.
5. Start the web app (Task 7):
   ```bash
   python app/app.py
   ```
   Open <http://127.0.0.1:5000> for the predictor and <http://127.0.0.1:5000/blog> for the blog (PDF at `/blog.pdf`).
   The app works from any folder. If a saved model is missing or was made by a different scikit-learn version, it rebuilds it from `data/detect_dataset.csv` on start-up. The page is a **real-time monitor**: the dataset is replayed sample by sample (adjustable speed, window, jump, "Find next fault"), every sample is classified instantly by both models, with live charts, prediction strips (actual / logistic / SVM) and running accuracy, missed-fault and false-alarm counts. The *Manual input* box re-predicts as you type, and the *Load real FAULT / NO-FAULT row* buttons fill it from the dataset. API: `GET /api/stream?start=0&n=300`, `POST /predict`.

`.vscode/launch.json` also has **Run and Debug** entries for `run_all.py` and the web app (press F5).

## Method notes

- The two empty columns and the label are never used as features.
- Mean/std come from the **training rows only**, also for Set Q; squares are computed from the **raw** values before standardising.
- α is the largest of {0.01, 0.1, 1, 10} for which J(θ) falls at every one of 2000 iterations (Set R → 1, Set Q → 10). C and γ are chosen on the validation set (ties go to the larger C).
- The test set is used once, at the end; each confusion matrix adds up to 2401.
- The SVM probability is not shown in the app: the saved `SVC` is not calibrated, so only its class and margin are meaningful.

## Reproducibility note

Everything reproduces the Colab results exactly, except tiny solver differences in the **linear SVM** (a different scikit-learn/libsvm version
gives validation accuracy 0.7258 instead of 0.7262 and 5836 instead of 5835 support vectors at C=100). Its test result is identical (73.64 %, 633 missed faults).
Load the saved `.joblib` files with a similar scikit-learn version to the one that created them.

## Team contributions

| Member | Work |
|---|---|
| Laiba Afzaal (23F-6073) | Tasks 1-4: data preparation and split, exploratory plots, logistic regression from scratch (Set R and Set Q), linear and RBF SVMs |
| Muhammad Ali (23F-6020) | Tasks 5-7: test-set evaluation (accuracy and confusion matrices), Task 6 analysis, real-time web front end and blog |
