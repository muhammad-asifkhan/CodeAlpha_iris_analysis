# Iris Flower Classification

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Accuracy](https://img.shields.io/badge/Best%20Accuracy-98%25-success)

> **CodeAlpha Data Science Internship — Task 1**  
> End-to-end multiclass classification pipeline with statistical hypothesis testing, dimensionality reduction, hyperparameter optimisation, and production-ready model persistence.

---

## Problem Statement

Given sepal and petal measurements of Iris flowers, accurately classify each sample into one of three species — *Iris setosa*, *Iris versicolor*, or *Iris virginica*. This is a canonical multiclass classification problem that benchmarks fundamental ML algorithms.

---

## Dataset

| Attribute | Detail |
|-----------|--------|
| Source | UCI Machine Learning Repository |
| Samples | 150 (50 per class — perfectly balanced) |
| Features | 4 numeric (Sepal Length, Sepal Width, Petal Length, Petal Width) |
| Target | Species (3 classes) |
| Missing Values | None |

---

## Methodology

```
Raw Data
   │
   ├── Exploratory Data Analysis
   │     ├── Distribution histograms & box plots per species
   │     ├── Violin plots (distribution shape)
   │     ├── Pearson correlation heatmap
   │     └── Pair plot (separability visualisation)
   │
   ├── Statistical Hypothesis Testing
   │     ├── Kruskal-Wallis test (per feature, H₀: equal medians)
   │     └── Pairwise Mann-Whitney U (Bonferroni corrected)
   │
   ├── Dimensionality Reduction
   │     └── PCA — 2D & Scree plot (97.8% variance in 2 PCs)
   │
   ├── Preprocessing
   │     ├── Drop Id column
   │     ├── Label encoding (target)
   │     ├── Stratified train/test split (80/20)
   │     └── StandardScaler
   │
   ├── Model Training & 5-Fold Stratified CV
   │     ├── Logistic Regression
   │     ├── K-Nearest Neighbors
   │     ├── SVM (RBF kernel)
   │     ├── Decision Tree
   │     ├── Random Forest
   │     └── Gradient Boosting
   │
   ├── Hyperparameter Tuning (GridSearchCV)
   │     ├── SVM: C ∈ {0.1, 1, 10, 100}, γ ∈ {scale, auto, 0.01, 0.1}
   │     └── RF: n_estimators, max_depth, min_samples_split
   │
   ├── Advanced Evaluation
   │     ├── ROC Curves (One-vs-Rest, AUC per class)
   │     ├── Learning Curves (bias-variance trade-off)
   │     ├── Confusion matrix
   │     ├── Permutation importance
   │     └── Classification report (Precision / Recall / F1)
   │
   └── Model Persistence (joblib Pipeline)
```

---

## Results

| Model | CV Accuracy | Test Accuracy |
|-------|:-----------:|:-------------:|
| Logistic Regression | 96.7% | 96.7% |
| K-Nearest Neighbors | 96.7% | 96.7% |
| SVM (RBF) | 97.5% | 96.7% |
| Decision Tree | 95.8% | 96.7% |
| Random Forest | 97.5% | 96.7% |
| Gradient Boosting | 97.5% | 96.7% |
| **SVM (Tuned — GridSearchCV)** | **98.3%** | **~100%** |

### Key Findings

- All 4 features are **highly statistically significant** across species (Kruskal-Wallis, p < 0.001).
- **Petal Length** and **Petal Width** contribute ~4× more information than sepal measurements.
- **PCA** confirms low intrinsic dimensionality — PC1 + PC2 capture **97.8%** of total variance.
- *Setosa* is **linearly separable**; Versicolor/Virginica require non-linear decision boundaries.
- **Learning curves** confirm no overfitting — training and CV scores converge by ~80 samples.
- Tuned **SVM (RBF)** achieves near-perfect classification with a compact model footprint.

---

## Project Structure

```
1_Iris_Analysis/
├── Iris_Classification.ipynb   # Full analysis notebook
├── data/
│   └── Iris.csv
├── models/
│   ├── iris_classifier.pkl     # Trained pipeline (scaler + classifier)
│   └── label_encoder.pkl
├── reports/
│   ├── 01_distributions.png
│   ├── 02_correlation.png
│   ├── 03_pairplot.png
│   ├── 04_pca.png
│   ├── 05_learning_curves.png
│   ├── 06_roc_curves.png
│   └── 07_final_evaluation.png
└── README.md
```

---

## How to Run

```bash
# Activate the shared virtual environment (from project root)
.\venv\Scripts\Activate.ps1       # Windows PowerShell
# source venv/bin/activate        # Linux / macOS

# Launch Jupyter
jupyter notebook 1_Iris_Analysis/Iris_Classification.ipynb
```

**Quick inference with saved model:**
```python
import joblib, numpy as np

pipeline = joblib.load("models/iris_classifier.pkl")
le       = joblib.load("models/label_encoder.pkl")

sample = np.array([[5.1, 3.5, 1.4, 0.2]])   # [SepalL, SepalW, PetalL, PetalW]
pred   = le.inverse_transform(pipeline.predict(sample))
print(pred[0])   # → Iris-setosa
```

---

## Technologies

| Library | Purpose |
|---------|---------|
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Visualisation |
| `scipy.stats` | Kruskal-Wallis, Mann-Whitney U |
| `sklearn.decomposition.PCA` | Dimensionality reduction |
| `sklearn.model_selection` | GridSearchCV, StratifiedKFold |
| `sklearn.inspection` | Permutation importance |
| `joblib` | Model persistence |

---

## Author

**Muhammad Asif Khan** — CodeAlpha Data Science Intern  
[GitHub](https://github.com) · [LinkedIn](https://linkedin.com)
