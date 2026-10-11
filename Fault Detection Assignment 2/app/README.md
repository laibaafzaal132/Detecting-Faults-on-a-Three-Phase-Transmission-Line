# Electrical Fault Detection — Task 7 Front End

A small Streamlit web interface for Assignment 2. It accepts `Ia`, `Ib`, `Ic`, `Va`, `Vb`, and `Vc`, then shows predictions side by side for:
- Logistic Regression on Set Q (12 features: six measurements plus their squares)
- RBF SVM (C=100, gamma=1), trained using the assignment's seeded split if the dataset is present

## Run in VS Code (Windows)

1. Extract this folder and open **AML_Fault_Detection_Frontend** in VS Code.
2. Open Terminal → New Terminal.
3. Create and activate a virtual environment (recommended):

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again.

4. Install libraries:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Start the interface:

   ```powershell
   streamlit run app/app.py
   ```

Streamlit opens the app in your browser. If it does not, open the local URL printed in the terminal (usually `http://localhost:8501`).

## Enabling the RBF SVM prediction

The notebook upload contains the learned Set-Q weights and training statistics, so Set-Q Logistic Regression works immediately. The notebook does not contain the actual trained RBF SVM binary artifact in this project folder, and the original CSV was not attached separately.

To enable RBF predictions:
1. Download `detect dataset.csv` from the Kaggle dataset specified in the assignment:
   https://www.kaggle.com/datasets/esathyaprakash/electrical-fault-detection-and-classification
2. Put the file at `data/detect dataset.csv` (keep the filename exactly as shown).
3. Restart the app. It automatically trains the RBF SVM with `C=100`, `gamma=1`, and saves it to `models/best_rbf_svm.joblib`.

The app deliberately does not make up RBF predictions when the required trained model/data is absent.

## Notes

- Input units must match the original CSV (currents in A, voltages in per unit).
- `0` means no fault and `1` means fault.
- This is an academic demonstration only, not a safety-certified relay or a substitute for a real protection system.
- The Set-Q coefficients and scaling statistics were copied from the provided Colab notebook outputs.
