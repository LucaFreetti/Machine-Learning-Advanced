# 🩺 Predicting Diabetes Progression

A supervised machine learning project to predict the 1-year progression of diabetes using clinical variables, built with scikit-learn.

---

## 📌 Objective

Predict the quantitative measure of diabetes progression one year ahead using 10 standardized clinical variables. This project simulates a real-world collaboration with a healthcare organization focused on **prevention and monitoring of chronic diseases**.

---

## 📊 Dataset

- **Source:** [`sklearn.datasets.load_diabetes`](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)
- **Samples:** 442 patients
- **Features:** 10 standardized numerical variables (mean = 0, variance = 1)
- **Target:** Quantitative measure of disease progression at 1 year

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Numeric | Age in years (standardized) |
| `sex` | Binary | Sex indicator (standardized) |
| `bmi` | Numeric | Body Mass Index — top predictor |
| `bp` | Numeric | Average blood pressure |
| `s1`–`s6` | Numeric | Blood serum indicators (cholesterol, triglycerides, etc.) |

---

## 🔍 Exploratory Data Analysis

Key findings from the correlation analysis:

- **`bmi` and `s5`** have the highest correlation with the target (~0.59) — strongest clinical predictors
- **`s1` and `s2`** are strongly correlated with each other (r ≈ 0.90), introducing multicollinearity — handled differently by Ridge vs GBR
- **`s3` (HDL cholesterol)** has a negative correlation with the target (-0.39) — higher HDL is protective
- **`sex` and `age`** show limited predictive power alone

> All 10 features were retained to ensure a fair model comparison and avoid losing information from a small dataset.

---

## ⚙️ Methodology

### Step 1 — Baseline: Ridge Regression
- Used `RidgeCV` with 5-fold cross-validation
- Alpha candidates: `[0.01, 0.1, 1, 10, 100]`
- Best alpha selected: **0.10** (minimal regularization needed — data already well-scaled)

### Step 2 — Main Model: Gradient Boosting Regressor
- Sequential ensemble of decision trees
- Each tree corrects the errors of the previous one

### Step 3 — Hyperparameter Tuning: GridSearchCV
- 5-fold cross-validation
- Scoring: `neg_mean_squared_error`
- Search space:

```python
params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [2, 3, 5],
    'min_samples_split': [2, 5]
}
```

### Step 4 — Evaluation
Metrics used: **R²**, **MAE**, **RMSE** on the held-out test set (25% split).

---

## 📈 Results

| Metric | Ridge (Baseline) | GBR (Tuned) |
|--------|-----------------|-------------|
| R² | **0.49** | ~0.47 |
| MAE | **41.55** | ~43.5 |
| RMSE | **53.01** | ~54.8 |

### Key Observations

- ✅ **Ridge Regression marginally outperforms GBR** on this dataset. This is expected: with only 442 samples and predominantly linear relationships, the simpler model generalizes better.
- ✅ **Clear positive trend** in predictions — the model has learned the relationship between features and the target.
- ✅ **Good performance in the central range** (100–200), where most patients are concentrated.
- ⚠️ **High variance for severe progressions** (230–310 range) — the model struggles to distinguish the most severe cases.
- ℹ️ An R² of ~0.49 is considered honest and realistic in clinical research, where patient variability (genetics, lifestyle, therapy) is not fully captured in the dataset.

> GBR's advantage typically emerges with larger samples and stronger non-linear patterns — neither of which is fully present here.

---

## 🏆 Feature Importance (GBR)

| Feature | Importance |
|---------|-----------|
| `bmi` (Body Mass Index) | ~39% |
| `s5` (Log-Triglycerides) | ~29% |
| `bp` (Blood Pressure) | ~13% |
| Other features | ~19% |

**Together, `bmi`, `s5`, and `bp` account for ~81% of the model's predictive power.**

---

## 🧰 Tech Stack

- Python 3.x
- [scikit-learn](https://scikit-learn.org/) — models, GridSearchCV, metrics
- [pandas](https://pandas.pydata.org/) — data manipulation
- [numpy](https://numpy.org/) — numerical operations
- [matplotlib](https://matplotlib.org/) & [seaborn](https://seaborn.pydata.org/) — visualizations

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/<your-username>/predicting-diabetes-progression.git
cd predicting-diabetes-progression

# Install dependencies
pip install scikit-learn pandas numpy matplotlib seaborn

# Run the notebook
jupyter notebook LucaFrittitta_Diabete_ML.ipynb
```

---

## 📁 Project Structure

```
.
├── LucaFrittitta_Diabete_ML.ipynb   # Full analysis notebook
└── README.md
```

---

## ✅ Clinical Takeaways

- 🏋️ **BMI is the single strongest predictor** (~39%) — high BMI patients should be prioritized for monitoring
- 🧪 **s5 (log-triglycerides) ranks second** (~29%) — elevated lipids signal accelerated insulin resistance
- 💓 **Blood pressure ranks third** (~13%) — consistent with cardiovascular complications in diabetes

> The model is suitable for identifying **general risk trends** (low / medium / high), but should not be used alone for individual clinical decisions without further validation.

---

## 👤 Author

**Luca Frittitta** — Machine Learning Project
