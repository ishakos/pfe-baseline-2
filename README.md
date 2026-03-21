# 🧠 LSTM Baseline for IoT Malware Detection

## 📌 Project Context

This work is part of a Master’s PFE project:

> **Malware Detection in IoT Systems using Deep Reinforcement Learning (DRL)**

Before implementing the DRL agent, we build **baseline models** to:

* establish performance benchmarks
* understand the dataset behavior
* validate preprocessing and data representation
* compare classical vs deep learning approaches

This repository contains the **second baseline model: an LSTM-based sequence model**.

---

## 🎯 Objective of This Model

Unlike the Random Forest baseline (tabular), this model aims to:

* capture **temporal patterns in network traffic**
* simulate **IoT device behavior over time**
* prepare data in a way that aligns with a **future DRL environment**

---

## 📊 Dataset

We use a **cleaned TON_IoT network dataset**:

* no missing values removed globally
* no encoding applied globally
* no scaling applied globally

👉 Important design:

> **Global dataset → Model-specific preprocessing**

Each model is responsible for its own preprocessing.

---

## ⚙️ Model Pipeline Overview

The LSTM pipeline follows these steps:

### 1. Data Loading

* Load `iot_dataset_clean.csv`
* Keep dataset unchanged

---

### 2. Model-Specific Preprocessing

Applied **only inside this model**:

* Numeric features:

  * Missing value imputation (median)
  * Standard scaling

* Categorical features:

  * Missing value imputation (most frequent)
  * One-hot encoding

👉 Preprocessing is:

* fitted on **training set only**
* applied to validation/test

---

### 3. Device-Based Splitting

Instead of random row splitting:

* Data is split using **`src_ip` (device identifier)**

This ensures:

* no leakage between devices
* realistic generalization to unseen devices

---

### 4. Sequence Construction (Key Contribution)

This is the most important part of the model.

We simulate IoT traffic streams by:

* grouping data by **device (`src_ip`)**
* creating **sliding windows**

Example:

```
Sequence length = 20

Flow 1 → Flow 20 → Sequence 1  
Flow 2 → Flow 21 → Sequence 2  
...
```

Final input shape:

```
(num_sequences, sequence_length, num_features)
```

👉 This transforms tabular data into **time-series data**

---

### 5. Sequence Labeling

Each sequence is assigned a label:

* using the **last element in the window**

```
sequence_label = label at time t (last timestep)
```

This models **current device state prediction**.

---

### 6. LSTM Architecture

The model is implemented using **PyTorch**:

```
Input (sequence)
   ↓
LSTM layers
   ↓
Last hidden state
   ↓
Dropout
   ↓
Fully Connected Layer
   ↓
Sigmoid (binary classification)
```

Key parameters:

* Hidden size: 128
* Layers: 2
* Dropout: 0.3

---

### 7. Training Strategy

* Loss: **Binary Cross Entropy with logits**
* Class imbalance handled using **pos_weight**
* Optimizer: Adam
* Early stopping based on **validation F1-score**

---

### 8. Evaluation Metrics

Same metrics as Random Forest baseline:

* Accuracy
* Precision
* Recall
* F1-score
* ROC AUC
* PR AUC

---

## 📈 Results

### Validation Performance

* F1-score: **0.9879**
* ROC AUC: **0.9899**

### Test Performance

* F1-score: **0.9363**
* ROC AUC: **0.9667**

---

## ⚠️ Important Observation

We observed a **strong class imbalance across splits**:

| Split      | Positive Ratio |
| ---------- | -------------- |
| Train      | 57.7%          |
| Validation | 97.0%          |
| Test       | 19.3%          |

👉 This happens because:

* data is split **by device**
* some devices contain mostly attack traffic
* others contain mostly normal traffic

### Impact

* Validation set is heavily biased toward attacks
* Test set is more balanced
* This causes a drop between validation and test performance

---

## 🧠 Interpretation of Results

* The model achieves **high precision (0.99)** → very few false positives
* Recall is lower (~0.88) → some attacks are missed
* This is typical for imbalanced anomaly detection

👉 Overall:

> The LSTM successfully learns temporal patterns and generalizes to unseen devices.

---

## 🔗 Relation to DRL (Very Important)

This model directly supports the DRL objective:

| LSTM Concept      | DRL Equivalent             |
| ----------------- | -------------------------- |
| Sequence (window) | State                      |
| Device (`src_ip`) | Environment / Agent source |
| Prediction        | Action (detect attack)     |
| Label             | Reward signal              |

👉 This pipeline prepares the data for:

```
IoT Device → Sequence → DRL Agent
```

---

## ✅ Key Contributions

* Transform tabular IoT data into **time-series sequences**
* Introduce **device-aware modeling**
* Build a **true LSTM baseline (not fake tabular)**
* Identify **data distribution issues across devices**
* Provide a structure compatible with **DRL integration**

---

## 🚀 Future Improvements

* Device-level **stratified splitting**
* Learn **categorical embeddings** instead of one-hot encoding
* Try different sequence labeling strategies:

  * majority vote
  * any-attack detection
* Use **bidirectional LSTM**
* Transition to **DRL agent (PPO / DQN)**

---

## 🧾 Conclusion

This LSTM model represents a **strong deep learning baseline** for IoT malware detection.

It improves over classical approaches by:

* incorporating **temporal dynamics**
* modeling **device-specific behavior**
* aligning the dataset with a **reinforcement learning framework**

It will serve as a **foundation for the upcoming DRL model**.

---
