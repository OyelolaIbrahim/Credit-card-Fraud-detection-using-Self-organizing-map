# Credit Card Fraud Detection — Self-Organizing Map (SOM)

Unsupervised anomaly detection on credit card applications 
using a 10×10 Self-Organizing Map to cluster customers and 
visually identify potentially fraudulent applications 
without using labelled fraud data during training.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Method](https://img.shields.io/badge/Method-Self--Organizing%20Map-purple)
![Task](https://img.shields.io/badge/Task-Anomaly%20Detection-red)
![Type](https://img.shields.io/badge/Type-Unsupervised%20Learning-green)

---

## Overview

Applies a Self-Organizing Map (SOM) to a credit card 
application dataset to detect anomalous customer profiles 
without relying on fraud labels during training. The SOM 
maps high-dimensional customer data onto a 2D grid where 
similar applications cluster together. A distance map 
(U-Matrix) is visualised to reveal cluster boundaries — 
nodes with high distances from their neighbours indicate 
outliers that are likely fraudulent applications.

---

## Dataset

- **Name:** Credit Card Applications Dataset
- **File:** `Credit_Card_Applications.csv`
- **Source:** [Download from Kaggle](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction)
- **Features used:** Columns 1–13 (customer application 
  features — demographics and financial information)
- **Target column:** Last column — approval status 
  (used only for visualisation overlay, not training)
- **Instructions:** Download the CSV and place it in 
  the root folder before running the notebook

---

## Approach

- Loaded dataset using Pandas; separated features 
  (`X` — columns 1–13) and labels (`y` — last column)
- Applied **MinMaxScaler** to normalise all features 
  to the [0, 1] range
- Trained a **10×10 MiniSom** grid with:
  - Input length: 13 features
  - Sigma: 1.0 (neighbourhood radius)
  - Learning rate: 0.5
  - Iterations: 20,000 random training steps
- Computed the **distance map (U-Matrix)** — measures 
  mean distance between each node and its neighbours
- Visualised the SOM with two plots:
  - Plain U-Matrix heatmap (`bone` + `pcolor` + `colorbar`)
  - Annotated map overlaying customer markers:
    - `o` red = approved application
    - `s` green = rejected application
  - High-distance (bright) nodes with red markers 
    indicate potentially fraudulent approved applications

---

## Results

| Parameter | Value |
|-----------|-------|
| SOM Grid | 10 × 10 |
| Training Iterations | 20,000 |
| Learning Rate | 0.5 |
| Sigma | 1.0 |
| Input Features | 13 |
| Normalisation | MinMaxScaler [0, 1] |

---

## Technologies Used

Python, MiniSom, Scikit-Learn (MinMaxScaler), 
NumPy, Pandas, Matplotlib (pylab)

---

## How to Run

```bash
git clone https://github.com/OyelolaIbrahim/fraud-detection-som.git
cd fraud-detection-som
pip install -r requirements.txt
jupyter notebook self_organizing_map.ipynb
```

---

## Key Takeaways

- SOMs learn without labels — the fraud pattern 
  emerges from the data structure itself
- Bright nodes on the U-Matrix indicate cluster 
  boundaries and outliers — these are the 
  high-risk application profiles
- Overlaying approval labels (`y`) after training 
  validates that the SOM correctly separates 
  suspicious profiles without ever seeing 
  those labels during training

---


