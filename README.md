# QEDL: Hybrid Quantum--Classical Deep Learning for Cyberattack Detection

A reference implementation of the **Quantum-Enhanced Deep Learning (QEDL)**
framework: a hybrid quantum--classical architecture for real-time
cyberattack detection, combining **QAOA-based quantum feature selection**,
**quantum data encoding**, a **parameterized quantum circuit (PQC)** front
end, and a **classical CNN/LSTM** back end.

This repository is a runnable, open-source implementation of the pipeline
described in the accompanying paper, *"A Hybrid Quantum-Classical Deep
Learning Framework for Real-Time Detection and Mitigation of Cyberattack
Patterns."* All quantum circuits are simulated (via
[PennyLane](https://pennylane.ai)'s `default.qubit` device); no physical
quantum hardware is required or used.

> **Note on fidelity to the paper.** The paper's QFS stage optimizes a QAOA
> cost Hamiltonian encoding classification error, using IBM Aer/Qiskit for
> simulation and PennyLane for circuit training. Directly optimizing a cost
> Hamiltonian over *retrained-classifier* loss is prohibitively expensive
> for a reference implementation, so this repo builds the cost Hamiltonian
> from a standard mRMR-style QUBO (feature relevance via mutual information,
> minus pairwise redundancy) -- a common practical proxy for exactly this
> kind of quantum feature-selection cost function. Everything downstream
> (angle encoding, hardware-efficient PQC ansatz, hybrid CNN/LSTM
> classifier, parameter-shift + backprop joint training) follows the paper directly.

## Architecture

```
Raw data --> Preprocess --> QAOA Feature Selection --> Angle Encoding --> PQC (HEA) --> CNN / LSTM --> Softmax
            (Stage 1)         (Stage 2, QFS)          (Stage 3, QDE)    (Stage 4)      (Stage 5)      (output)
```

| Stage | Module | Description |
|---|---|---|
| 1 | `src/preprocessing.py` | Normalization, one-hot encoding, mean imputation, stratified train/val/test split |
| 2 | `src/quantum_feature_selection.py` | QAOA over a QUBO cost/mixer Hamiltonian to select a reduced, non-redundant feature subset |
| 3-4 | `src/quantum_encoding.py` | Angle (or amplitude) encoding + hardware-efficient PQC ansatz (`R_y`, `R_z`, ring-topology CNOTs), exposed as a `pennylane.qnn.KerasLayer` |
| 5 | `src/models.py` | Hybrid classifier: PQC output feeds a CNN (spatial data) or LSTM (sequential data) head |
| 6 | `src/train.py` | Joint training: classical layers via backprop, PQC layer via the parameter-shift rule, both inside one Adam optimizer step; 10-fold cross-validation |
| 7 | `src/evaluate.py` | Accuracy, Precision, Recall, F1-score, per-sample inference time |

## Installation

```bash
git clone <your-repo-url>.git
cd qedl-framework
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

Tested with Python 3.10/3.11, TensorFlow 2.13-2.15, and PennyLane 0.36.

## Data

Place a CIC-IDS2017 / UNSW-NB15 (or any similarly structured labeled flow
dataset) CSV file at the path given in `config.yaml` (`data.path`), e.g.:

```
data/cicids2017.csv
```

Download CIC-IDS2017 from the Canadian Institute for Cybersecurity:
https://www.unb.ca/cic/datasets/ids-2017.html

Update `data.target_column` in `config.yaml` to match your label column
name (e.g. `"Label"`), and `model.num_classes` is inferred automatically
from the number of unique labels found.

## Usage

Run the full pipeline (QFS -> QDE/PQC -> hybrid classifier -> 10-fold CV):

```bash
python main.py --config config.yaml
```

Run ablation studies (matching Section 7 of the paper):

```bash
# remove quantum feature selection
python main.py --config config.yaml --ablate-qfs

# remove the PQC / quantum encoding front end
python main.py --config config.yaml --ablate-pqc
```

Results (per-fold and aggregated mean +/- std metrics) are written to
`results/kfold_summary.json`; the trained model is saved to
`results/saved_model/qedl_model.keras`.

## Configuration

All hyperparameters live in `config.yaml`:

- `qfs`: QAOA search-space size, target number of selected features, QAOA depth/steps/learning rate
- `pqc`: PQC register size, ansatz depth (`L`), encoding strategy (`angle` / `amplitude`)
- `model`: CNN vs. LSTM back end, layer sizes, dropout
- `train`: batch size, epochs, learning rate, early-stopping patience, number of CV folds

## Repository structure

```
qedl-framework/
├── config.yaml
├── main.py
├── requirements.txt
├── data/                       # place your dataset CSV here (not tracked by git)
├── results/                    # metrics + saved model (not tracked by git)
└── src/
    ├── preprocessing.py
    ├── quantum_feature_selection.py
    ├── quantum_encoding.py
    ├── models.py
    ├── train.py
    └── evaluate.py
```

## Extending this repo

- **Amplitude encoding**: set `pqc.encoding: "amplitude"` in `config.yaml`.
- **Different attack taxonomies**: adjust `model.num_classes` / your label column; it's inferred automatically at runtime.
- **Physical quantum hardware**: swap `qml.device("default.qubit", ...)` in `quantum_encoding.py` / `quantum_feature_selection.py` for a PennyLane hardware plugin (e.g. `qiskit.ibmq`, `braket.aws`); note this will significantly change runtime and require shot-based (rather than exact statevector) expectation values.

## Known issues / warnings

- **PennyLane TF-interface deprecation warning.** Recent PennyLane releases
  are shifting their machine-learning-framework focus to JAX/PyTorch, and
  print a `PennyLaneDeprecationWarning` when using `interface="tf"`. The
  code here still runs correctly under this interface as of PennyLane
  0.38-0.45; if a future release removes TF support outright, port
  `src/quantum_encoding.py`'s `PQCLayer` to JAX (`interface="jax"`) with a
  small `jax2tf` bridge, or move the whole classical back end to PyTorch
  and use PennyLane's still-supported `TorchLayer`.
- **`run_eagerly=True`.** The PQC layer requires eager execution because
  PennyLane's TF interface does not support Keras's symbolic graph tracing.
  This is why the model is built with the Keras *subclassing* API (not the
  Functional API) and compiled with `run_eagerly=True` -- expect somewhat
  slower training than a purely classical Keras model, dominated by
  statevector-simulation cost.
- **Custom-layer serialization.** Because of the above, the trained model
  is saved/restored via `model.save_weights(...)` / `model.load_weights(...)`
  rather than a full `.keras` archive -- rebuild the architecture from
  `config.yaml` via `build_hybrid_model(...)` before loading weights.

## Citation

If you use this code, please cite the accompanying paper:

```bibtex
@article{khan2026qedl,
  title   = {A Hybrid Quantum-Classical Deep Learning Framework for Real-Time Detection and Mitigation of Cyberattack Patterns},
  author  = {Khan, Nayeem Ahmad and Alarfaj, Fawaz Khaled},
  year    = {2026}
}
```

## License

MIT (or your preferred license -- update this section before publishing).
