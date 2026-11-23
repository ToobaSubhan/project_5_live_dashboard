import copy
import torch
from torch import nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(scale=True):
    d = load_breast_cancer()
    X = d.data.astype("float32")
    y = (1 - d.target).astype("float32")  # sklearn has 1 = benign; flip so 1 = malignant
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=0)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=0)
    if scale:
        sc = StandardScaler().fit(X_tr)  # fitted on training rows only
        X_tr, X_va, X_te = (sc.transform(a).astype("float32") for a in (X_tr, X_va, X_te))
    t = torch.tensor
    return {"X_tr": t(X_tr), "y_tr": t(y_tr), "X_va": t(X_va), "y_va": t(y_va),
            "X_te": t(X_te), "y_te": t(y_te)}


def make_model(units, dropout, n_in=30):
    return nn.Sequential(
        nn.Linear(n_in, units), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(units, units), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(units, 1),
    )


def train(cfg, data, on_epoch=None):
    """cfg keys: units, dropout, decay, lr, epochs, early (bool), patience."""
    torch.manual_seed(0)
    model = make_model(cfg["units"], cfg["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["decay"])
    lossf = nn.BCEWithLogitsLoss()
    X, y = data["X_tr"], data["y_tr"]
    hist = {"train": [], "val": []}
    best, best_state, wait = float("inf"), None, 0

    for epoch in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 32):
            idx = perm[i:i + 32]
            opt.zero_grad()
            lossf(model(X[idx]).squeeze(1), y[idx]).backward()
            opt.step()

        model.eval()  # losses measured in eval mode so dropout is off for both curves
        with torch.no_grad():
            tr = lossf(model(X).squeeze(1), y).item()
            va = lossf(model(data["X_va"]).squeeze(1), data["y_va"]).item()
        hist["train"].append(tr)
        hist["val"].append(va)

        if va < best:
            best, best_state, wait = va, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1

        if on_epoch:
            on_epoch(epoch, hist)
        if cfg["early"] and wait >= cfg["patience"]:
            break

    if cfg["early"]:
        model.load_state_dict(best_state)  # keep the best model, not the last one
    return model, hist


def diagnose(hist):
    if hist["train"][-1] > 0.3:
        return "underfitting"
    if hist["val"][-1] > 1.3 * min(hist["val"]):
        return "overfitting"
    return "healthy"


def evaluate(model, X, y, thr=0.5):
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(X).squeeze(1))
    pred = (p >= thr).float()
    found = int(((pred == 1) & (y == 1)).sum())
    total = int((y == 1).sum())
    false_alarms = int(((pred == 1) & (y == 0)).sum())
    return found, total, false_alarms, p