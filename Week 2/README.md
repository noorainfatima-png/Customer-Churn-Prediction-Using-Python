# Data Engineering & Preprocessing Plan — Telco Customer Churn (Week 2)

**A technical execution plan for acquiring, validating, cleaning, transforming, and partitioning the IBM Telco Customer Churn dataset using Python.**

---

## Executive Summary

This document defines the **Week 2 Data Engineering and Preprocessing Strategy** for the Telco Customer Churn project. Moving from Week 1 planning to operational execution, this phase establishes a reproducible Python pipeline to prepare raw customer records for machine learning. The pipeline addresses string blanks, incorrect data types, categorical encoding, scaling, and stratified train-test partitioning to ensure feature matrices are fully optimized for downstream classification models.

---

## Dataset Overview & Acquisition

* **Dataset:** Telco Customer Churn (IBM Sample Data via Kaggle)
* **Location:** `Week-2/data/WA_Fn-UseC_-Telco-Customer-Churn.csv`
* **Volume:** 7,043 rows $\times$ 21 columns
* **Target Label:** `Churn` (`Yes`: 1,869 | `No`: 5,174)

This dataset provides an ideal benchmark due to its real-world telecom features—spanning demographics, account details, and usage metrics—and manageable size for rapid iterative modeling on local hardware.

---

## Preprocessing Pipeline & Methodology

```text
[ Raw CSV Data ] ➡️ [ Inspection & Type Cast ] ➡️ [ Impute TotalCharges ]
                                                          │
[ Stratified Split ] ⬅️ [ Standard Scaler ] ⬅️ [ One-Hot Encoding ]

```

1. **Ingestion & Validation:** Ingest the CSV with `pandas`. Verify structure ($7043 \times 21$) and check for duplicates using `df.duplicated().sum()`.
2. **Data Cleaning:** `TotalCharges` parses as an object string due to 11 empty whitespace values in new accounts (`tenure` = 0). Convert these to `NaN` using `pd.to_numeric(errors='coerce')` and impute with `0.0`.
3. **Encoding:** Convert binary attributes (`Partner`, `Dependents`) to binary flags. Apply `pd.get_dummies()` for multi-class categories (`Contract`, `PaymentMethod`).
4. **Scaling & Stratification:** Standardize continuous features (`tenure`, `MonthlyCharges`, `TotalCharges`) using `StandardScaler`. Execute an 80/20 train-test split stratified on `Churn` to preserve target distribution.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = pd.get_dummies(df.drop(columns=["customerID", "Churn"]), drop_first=True)
y = df["Churn"].map({"Yes": 1, "No": 0})

X[["tenure", "MonthlyCharges", "TotalCharges"]] = StandardScaler().fit_transform(
    X[["tenure", "MonthlyCharges", "TotalCharges"]]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

```

---

## Technical Stack

* **Pandas:** Data loading, structural inspection, string trimming, missing value handling, and one-hot encoding.
* **NumPy:** Vectorized calculations and numerical array handling.
* **Scikit-learn:** Feature scaling (`StandardScaler`) and reproducible stratified partitioning (`train_test_split`).
