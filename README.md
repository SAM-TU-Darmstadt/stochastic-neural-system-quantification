# Stochastic Neural System Quantification (SNSQ)

**Stochastic Data Generation and Adaptive Forecasting in Additive Manufacturing using Neural Networks**

---

## 🌍 Overview

This repository presents **Stochastic Neural System Quantification (SNSQ)**, a novel framework designed to enable precise, adaptive prediction in dynamic and uncertain process environments, with a focus on **additive manufacturing**.

SNSQ is based on two core principles:

- **Knowledge Transfer**: Efficient propagation of known and unknown process knowledge.
- **System Quantification**: Associating behavioral vectors with observations to generate predictive functions.

Unlike static machine learning models, SNSQ allows for real-time adaptation and has shown strong results in complex manufacturing scenarios.

---

## ⚙️ SDNN – Stochastic Data Generation using Neural Networks

`SDNN` is the core technical implementation of SNSQ, enabling stochastic generation and behavioral modeling via neural networks.

---

## 🚀 TL;DR Quickstart

1. Place your data in the `exp_data/` directory.
2. Configure parameters in `config.py`.
3. Run the following scripts in order:
   - `pinn_train.py` – Train the physics-informed neural network
   - `sdnn_go.py` – Launch the SDNN process
   - `train_rnn.py` – Train the recurrent network for behavior approximation
   - `train_ae.py` – Train the autoencoder to define the behavioral vector (BV)

---

## 🧠 Architecture & Modules

```
stochastic-neural-system-quantification/
├── snsq/               # Core SNSQ components
│   ├── models/         # PINNs, RNNs, AEs
│   ├── utils/          # Utility functions
├── config.py           # Configuration file
├── exp_data/           # Input experimental data
├── sdnn_go.py          # Launches stochastic data generation
├── pinn_train.py       # Physics-informed training
├── train_ae.py         # Autoencoder training for BV
├── train_rnn.py        # RNN training for BV approximation
├── f_predict.py        # Prediction script using trained models
├── requirements.txt
└── README.md
```

---

## 🧪 First: PINN Training

### 1. Configure Parameters

Set global and local training parameters in `config.py` using `global_config` and related sections.

### 2. Input Pretraining Data

Place your physical/experimental data in the `exp_data/` directory.

### 3. Define Network Architecture

Modify the `set_architecture()` function in `config.py` to define your neural network structure.

### 4. Train the PINN

```shell
python pinn_train.py
```

Optimize hyperparameters via `config.py`.

---

## 🌪️ Stochastic Data Generation with SDNN

### 1. Configure SDNN

Modify the `sdnn_config()` function in `config.py` to define your generation parameters.

### 2. Launch the Process

```shell
python sdnn_go.py
```

---

## 🧬 Behavioral Vector (BV) Modeling

### 1. Train Autoencoder

Adjust `config.py` and run:

```shell
python train_ae.py
```

The autoencoder defines and saves the behavioral vector (BV).

### 2. Train Recurrent Network (RNN)

Train the RNN to approximate the BV from new data:

```shell
python train_rnn.py
```

---

## 🔮 Final Prediction Model (F)

After integrating the real data and the BV, use:

```shell
python f_predict.py
```

Adjust `input_shape` to reflect the inclusion of the BV.

---

## 🐍 Python Environment Setup (Recommended)

### Using Conda

```shell
conda create -n tensorflow_env tensorflow
conda activate tensorflow_env
conda install -c anaconda jupyter spyder tensorflow openpyxl
conda install -c conda-forge matplotlib pathlib scipy pydoe scikit-learn pandas
spyder
```

### Using Virtualenv

```shell
python -m venv tf_env
tf_env\scripts\activate
pip install spyder tensorflow matplotlib pathlib scipy pydoe scikit-learn openpyxl pandas
spyder
```

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for more information.

---

## 🤝 Contributors

- Sören Wenzel – Concept, Research, and Implementation
- Laura Sun - Testing and Implementation

---

## 📫 Contact

If you'd like to collaborate, ask questions, or discuss the project:

- 📧 Email: soeren.wenzel@tu-darmstadt.de
- 🧑‍💻 GitHub: [soren-wenzel](https://github.com/soren-wenzel)
