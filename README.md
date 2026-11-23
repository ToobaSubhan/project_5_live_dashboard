# Breast Cancer Training Dashboard

A Streamlit classroom demo for interactively training a binary classifier, inspecting training curves, and exploring how a decision threshold affects screening metrics. It is for education only; it is not a medical device and must not be used for clinical decisions.

## Objectives

- Demonstrate how neural-network hyperparameters affect training and validation loss.
- Visualize training progress and provide a simple underfitting/overfitting heuristic.
- Show the trade-off between sensitivity (recall) and false alarms as the classification threshold changes.
- Keep preprocessing leakage out of the evaluation by fitting the scaler on training data only.

## Project contents

```text
.
|-- app.py
|-- trainer.py
|-- requirements.txt
|-- requirements-trainer.txt
|-- .gitignore
`-- README.md
```

- `app.py` is the interactive TensorFlow/Keras Streamlit application.
- `trainer.py` contains a separate PyTorch training, diagnosis, and evaluation helper. It is not called by the Streamlit app.
- `requirements.txt` contains dependencies for the dashboard.
- `requirements-trainer.txt` contains the additional dependencies for `trainer.py`.

The breast-cancer dataset is loaded through scikit-learn at runtime; no dataset file needs to be downloaded or committed.

## Implementation

The dashboard loads scikit-learn's Wisconsin Diagnostic Breast Cancer dataset. The original target is inverted so that `1` means malignant (the positive/disease class) and `0` means benign. It uses stratified, reproducible 60/20/20 training, validation, and test splits. A `StandardScaler` is fitted on the training partition and applied to the other partitions.

The app's configurable Keras model consists of a selected number of fully connected ReLU layers, optional dropout and L2 regularization, and a sigmoid output. It is trained with Adam and binary cross-entropy. Live loss and accuracy charts are updated each epoch. Optional early stopping restores the best validation-loss weights. The training-curve diagnosis is a heuristic, not a statistical guarantee.

On the screening tab, the trained model predicts the held-out test partition. A selectable threshold determines positive predictions; the app reports recall, precision, F1, and a confusion matrix. It also lets a user inspect an individual held-out row and shows its known label for demonstration purposes.

The standalone PyTorch helper uses the same dataset and split proportions, but a different random seed and a fixed two-hidden-layer architecture. It exposes training, a simple curve diagnosis, and an evaluation helper; its behavior and results are separate from the Streamlit dashboard.

## Setup

Use Python 3.10 or 3.11 for broad compatibility with the TensorFlow dependency. From the repository directory in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal and activate the environment again.

To run the dashboard:

```powershell
streamlit run app.py
```

To use the standalone PyTorch helper, install its optional dependencies as well:

```powershell
python -m pip install -r requirements-trainer.txt
```

`trainer.py` exports `load_data`, `make_model`, `train`, `diagnose`, and `evaluate`; import and call these from a Python script or notebook. It does not provide a command-line interface.

## Using the dashboard

1. Choose model and training settings in the sidebar.
2. Select **Train** and inspect the live loss and accuracy curves.
3. Use the screening tab to evaluate the trained model on the held-out test set.
4. Adjust the decision threshold. A lower threshold generally catches more positive cases, while potentially increasing false positives.
5. Inspect a test row to see a demo probability and its known dataset label.

## Results

The application computes metrics when a model is trained and the held-out evaluation is requested. Exact scores depend on the selected hyperparameters and threshold, so this repository does not claim a fixed benchmark result. The app displays the measured test-set recall, precision, F1, and confusion matrix directly.

## Limitations

- This is a classroom demonstration using a small, public, preprocessed tabular dataset; it is not representative evidence of clinical performance.
- The displayed probability is a model score and is not established to be a calibrated clinical risk.
- A threshold, model, or metric from this demo must not guide diagnosis, treatment, or patient care.
- The individual-row view reveals the known test label and is for teaching only; it is not an appropriate workflow for real screening.
- Results can vary with hyperparameters. A single held-out split does not establish generalization across populations, institutions, or devices.
- The curve diagnosis is rule-based and can mischaracterize a model's fit.
- The PyTorch helper and Keras app are separate implementations with different seeds and model details; their results are not directly comparable.

## License

No license is currently specified. Add a license before redistributing this project if you intend to grant reuse rights.
