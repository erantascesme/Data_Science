# Introduction to Data Science — Final Assignment

This repository contains my final assignment for an **Introduction to Data Science** course, organized into three Jupyter notebooks:

- **Part A — Supervised Learning:** house price prediction (regression)
- **Part B — Unsupervised Learning:** clustering with DBSCAN on two datasets
- **Part C — Word Embedding:** sentiment classification with TF‑IDF vs. Word2Vec (and a hybrid model)


## Part A — Supervised Learning (Regression)

**Notebook:** `Supervised Learning.ipynb`  
**Goal:** Predict `SalePrice` for unseen houses in the test set.

### Workflow
- **Initial exploration & EDA:** variable types (numeric / categorical / ordinal), distributions, missingness patterns.
- **Missing values:** different strategies by feature type and missingness rate (mode/median, KNN-based imputation for some categorical/ordinal, and MICE/iterative imputation for numeric features).
- **Outliers:** data cleaning and anomaly removal (e.g., invalid `SalePrice`, suspicious year values), plus model-based detection using **Isolation Forest** and **Local Outlier Factor**, visualized in 2D with **PCA**.
- **Feature engineering:** transformations + encoding (e.g., one-hot, frequency encoding), interaction features, and scaling when required.
- **Feature selection:** variance filtering + **RFE** to reduce dimensionality and mitigate overfitting.
- **Multicollinearity handling:** correlation analysis and reduction of redundant features.
- **Model training & tuning:** compared multiple regressors and tuned hyperparameters **XGBoost, RandomForest, ElasticNet** (Bayesian optimization via `skopt`).
- **Explainability (XAI):**  
  - **Global:** **SHAP** summary + bar plots  
  - **Local:** **LIME** explanations for individual predictions


## Part B — Unsupervised Learning (DBSCAN)

**Notebook:** `Unsupervised Learning.ipynb`  
**Goal:** Cluster two unlabeled datasets using **DBSCAN**, including parameter search and quantitative evaluation.

### Workflow
- Standardization with **StandardScaler**
- Grid search over:
  - `eps` (0.1 → 2.0)
  - `min_samples` (3 → 9)
  - distance metrics: `euclidean`, `manhattan`, `chebyshev`
- Evaluation metrics:
  - **Silhouette score**
  - **Davies–Bouldin index**
  - **Calinski–Harabasz index**
- Visualization via **PCA** and **t‑SNE**


## Part C — Word Embedding (Sentiment Classification)

**Notebook:** `Word Embedding.ipynb`  
**Goal:** Classify review sentiment using different text representations and compare them.

### Workflow
- Text preprocessing (tokenization, stopword removal, cleaning)
- Vocabulary analysis + exploratory statistics
- Models:
  1. **TF‑IDF + Logistic Regression**
  2. **Word2Vec + Logistic Regression** (with parameter tuning)
  3. **Hybrid model:** concatenated TF‑IDF and Word2Vec features
