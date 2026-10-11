import sys
from pathlib import Path
import numpy as np
import streamlit as st
import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.model_utils import (
    Q_THETA, Q_MEAN, Q_STD, FEATURE_NAMES,
    predict_set_q, load_or_train_rbf, DATA_PATH
)

st.set_page_config(page_title="Electrical Fault Detection", page_icon="⚡", layout="wide")
st.title("⚡ Electrical Transmission-Line Fault Detection")
st.caption("Enter one instant of three-phase currents and voltages. Predictions are produced by Set-Q Logistic Regression and an RBF SVM.")

with st.sidebar:
    st.header("Input measurements")
    st.caption("Use the same units as the assignment dataset: current in A and voltage in per unit.")
    with st.form("measurements"):
        c1, c2 = st.columns(2)
        with c1:
            ia = st.number_input("Ia (A)", value=0.0, format="%.6f")
            ib = st.number_input("Ib (A)", value=0.0, format="%.6f")
            ic = st.number_input("Ic (A)", value=0.0, format="%.6f")
        with c2:
            va = st.number_input("Va (p.u.)", value=0.0, format="%.6f")
            vb = st.number_input("Vb (p.u.)", value=0.0, format="%.6f")
            vc = st.number_input("Vc (p.u.)", value=0.0, format="%.6f")
        submitted = st.form_submit_button("Predict fault", type="primary", use_container_width=True)

st.info("**Output meaning:** 0 = No fault, 1 = Fault. This is an educational assignment model, not certified protection equipment.")

if submitted:
    raw = np.array([ia, ib, ic, va, vb, vc], dtype=float)
    q_pred, q_prob = predict_set_q(raw)
    st.subheader("Prediction results")
    left, right = st.columns(2)
    with left:
        st.markdown("### Set-Q Logistic Regression")
        st.metric("Prediction", "FAULT (1)" if q_pred else "NO FAULT (0)")
        st.metric("Estimated fault probability", f"{q_prob*100:.2f}%")
        st.caption("Uses the 12-feature model (six raw measurements plus their squares), with training statistics and θ taken from the submitted notebook.")
    with right:
        st.markdown("### RBF Support Vector Machine")
        try:
            bundle = load_or_train_rbf()
            x_mean = np.asarray(bundle["mean"], dtype=float)
            x_std = np.asarray(bundle["std"], dtype=float)
            model = bundle["model"]
            x_scaled = (raw - x_mean) / np.where(x_std == 0, 1, x_std)
            rbf_pred = int(model.predict(x_scaled.reshape(1, -1))[0])
            st.metric("Prediction", "FAULT (1)" if rbf_pred else "NO FAULT (0)")
            st.caption(f"Model: RBF SVM, C={bundle.get('C', 100)}, gamma={bundle.get('gamma', 1)}.")
            if rbf_pred != q_pred:
                st.warning("The two models disagree for these measurements. Review the readings and treat this result as uncertain.")
        except Exception as e:
            st.warning("The RBF SVM model is not available yet.")
            st.write("To enable it, place **detect dataset.csv** in the project's `data` folder, then restart the app. The app will train and save the RBF model automatically.")
            st.caption(f"Details: {e}")

    with st.expander("Show transformed feature values"):
        q_raw = np.concatenate([raw, raw**2])
        q_scaled = (q_raw - Q_MEAN) / np.where(Q_STD == 0, 1, Q_STD)
        st.write("Standardized Set-Q features (before adding intercept):")
        st.dataframe({"Feature": FEATURE_NAMES + [f"{x}²" for x in FEATURE_NAMES],
                      "Standardized value": q_scaled}, use_container_width=True)

st.divider()
st.subheader("How to use")
st.markdown("""
1. Enter the six instantaneous measurements in the left panel.
2. Click **Predict fault**.
3. Compare the Set-Q Logistic Regression and RBF SVM predictions.
4. A prediction of **1** indicates a detected fault; **0** indicates no fault.
""")
st.caption("Model source: Assignment 2 notebook outputs. Predictions depend on the dataset units and preprocessing used during training.")
