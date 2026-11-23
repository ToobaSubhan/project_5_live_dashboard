import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="City Hospital - Live Training", layout="wide")
st.title("City Hospital: live training dashboard")
st.warning("Classroom demo only. Not a medical device and not for clinical use.")

# ---------- Data ----------
@st.cache_data
def load_data():
    d = load_breast_cancer(as_frame=True)
    X, y = d.data, d.target          # sklearn: 1 = benign, 0 = malignant
    y = 1 - y                         # flip so 1 = disease (malignant)
    return X, y

X, y = load_data()
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

scaler = StandardScaler().fit(X_tr)   # fit on train only
Xs_tr, Xs_val, Xs_te = (scaler.transform(a) for a in (X_tr, X_val, X_te))

# ---------- Sidebar controls ----------
st.sidebar.header("Hyperparameters")
units = st.sidebar.select_slider("Hidden units", [4, 8, 16, 32, 64, 128], value=32)
layers = st.sidebar.slider("Hidden layers", 1, 4, 2)
dropout = st.sidebar.slider("Dropout", 0.0, 0.7, 0.0, 0.05)
l2 = st.sidebar.select_slider("L2 weight decay", [0.0, 1e-5, 1e-4, 1e-3, 1e-2], value=0.0)
lr = st.sidebar.select_slider("Learning rate", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2], value=1e-3)
epochs = st.sidebar.slider("Max epochs", 10, 200, 60)
batch = st.sidebar.select_slider("Batch size", [8, 16, 32, 64], value=32)
early = st.sidebar.checkbox("Early stopping (patience 8)", value=False)

def build():
    reg = tf.keras.regularizers.l2(l2) if l2 > 0 else None
    m = tf.keras.Sequential([tf.keras.layers.Input(shape=(Xs_tr.shape[1],))])
    for _ in range(layers):
        m.add(tf.keras.layers.Dense(units, activation="relu", kernel_regularizer=reg))
        if dropout > 0:
            m.add(tf.keras.layers.Dropout(dropout))
    m.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy",
              metrics=["accuracy", tf.keras.metrics.Recall(name="recall")])
    return m

def diagnose(h):
    tl, vl = h["loss"], h["val_loss"]
    gap = vl[-1] - tl[-1]
    val_rising = len(vl) > 10 and vl[-1] > min(vl) * 1.10
    if tl[-1] > 0.35 and vl[-1] > 0.35:
        return "Likely UNDERFITTING: both losses are still high. Add capacity, raise the learning rate, or train longer."
    if gap > 0.15 or val_rising:
        return "Likely OVERFITTING: validation loss is far above (or rising away from) training loss. Add dropout/L2, shrink the model, or use early stopping."
    return "Curves look reasonably fit. Confirm on the screening tab."

# ---------- Live training ----------
class LiveCB(tf.keras.callbacks.Callback):
    def __init__(self, loss_ph, acc_ph, status):
        super().__init__()
        self.loss_ph, self.acc_ph, self.status = loss_ph, acc_ph, status
        self.hist = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
    def on_epoch_end(self, epoch, logs=None):
        for k in self.hist:
            self.hist[k].append(logs[k])
        self.loss_ph.line_chart(pd.DataFrame({"train": self.hist["loss"], "val": self.hist["val_loss"]}))
        self.acc_ph.line_chart(pd.DataFrame({"train": self.hist["accuracy"], "val": self.hist["val_accuracy"]}))
        self.status.write(f"Epoch {epoch + 1}/{epochs}")

tab_train, tab_screen = st.tabs(["1. Train & fix curves", "2. Screen patients"])

with tab_train:
    if st.button("Train", type="primary"):
        tf.keras.utils.set_random_seed(42)
        model = build()
        c1, c2 = st.columns(2)
        c1.subheader("Loss"); c2.subheader("Accuracy")
        loss_ph, acc_ph, status = c1.empty(), c2.empty(), st.empty()
        cbs = [LiveCB(loss_ph, acc_ph, status)]
        if early:
            cbs.append(tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True))
        model.fit(Xs_tr, y_tr, validation_data=(Xs_val, y_val),
                  epochs=epochs, batch_size=batch, verbose=0, callbacks=cbs)
        st.session_state["model"] = model
        st.session_state["hist"] = cbs[0].hist
        st.session_state.pop("test_done", None)
    if "hist" in st.session_state:
        st.info(diagnose(st.session_state["hist"]))
    else:
        st.caption("Set hyperparameters in the sidebar, then press Train.")

# ---------- Screening ----------
with tab_screen:
    if "model" not in st.session_state:
        st.caption("Train a model first.")
    else:
        model = st.session_state["model"]
        thr = st.slider("Decision threshold (lower = catch more disease, more false alarms)",
                        0.05, 0.95, 0.5, 0.05)
        if st.button("Evaluate on held-out test set"):
            st.session_state["test_done"] = True
        if st.session_state.get("test_done"):
            p = model.predict(Xs_te, verbose=0).ravel()
            pred = (p >= thr).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
            m1, m2, m3 = st.columns(3)
            m1.metric("Recall (sensitivity)", f"{recall_score(y_te, pred):.3f}")
            m2.metric("Precision", f"{precision_score(y_te, pred, zero_division=0):.3f}")
            m3.metric("F1", f"{f1_score(y_te, pred, zero_division=0):.3f}")
            st.write(pd.DataFrame([[tn, fp], [fn, tp]],
                     index=["Actual healthy", "Actual disease"],
                     columns=["Pred healthy", "Pred disease"]))
            st.caption(f"Missed cases (false negatives): {fn} of {fn + tp} diseased patients in the test set.")

        st.subheader("Screen one patient")
        idx = st.number_input("Test patient row", 0, len(X_te) - 1, 0)
        row = X_te.iloc[[int(idx)]]
        prob = float(model.predict(scaler.transform(row), verbose=0)[0, 0])
        st.progress(min(max(prob, 0.0), 1.0))
        st.write(f"Estimated disease probability: **{prob:.2%}**")
        st.write("Result: **REFER for follow-up**" if prob >= thr else "Result: no flag raised")
        st.caption(f"Actual label (for demo only): {'disease' if y_te.iloc[int(idx)] == 1 else 'healthy'}")