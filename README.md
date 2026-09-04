# Customer Churn Prediction & Analytics Pipeline

**An end-to-end machine learning project to identify, analyze, and predict customer churn using Python, data engineering best practices, and advanced classification modeling.**

---

## Project Overview

Customer churn directly impacts long-term Customer Lifetime Value (CLV). This project delivers a comprehensive, multi-week data science framework—moving from strategic planning and technical data engineering to predictive modeling and deployment strategy. By leveraging the **IBM Telco Customer Churn** and **UCI Online Retail** datasets, this pipeline translates raw transactional and customer profile data into actionable business intelligence for retention strategies.

---

## Repository Navigation

* **Week 1:** Strategic Planning & Business Objectives Framework
* **Week 2:** Data Acquisition, Validation, Cleaning & Feature Preprocessing Strategy
* **Week 3:** Exploratory Data Analysis (EDA) & Feature Correlation Analysis
* **Week 4:** Predictive Analytics Modeling Strategy, Hyperparameter Tuning & Deployment

---

## Key Technical Modules

* **Data Engineering Pipeline:** Ingestion, whitespace-to-null conversion, numerical type coercion (`TotalCharges`), median imputation, and de-duplication.
* **Feature Engineering & Transformation:** Multi-class One-Hot Encoding (`get_dummies`), binary encoding, and zero-mean feature standardization using `StandardScaler`.
* **Validation & Splitting:** Stratified 80/20 train-test partitioning alongside $k$-fold cross-validation to maintain target balance and prevent data leakage.
* **Predictive Modeling:** Multi-stage evaluation progressing from interpretable baselines (**Logistic Regression**) to non-linear trees (**Decision Trees**) and production-grade ensembles (**Random Forest**, **Gradient Boosting**).
* **Metrics Strategy:** Metric optimization prioritized around **Recall** and **F1-Score** over accuracy to minimize costly false negatives.

---

## Core Dataset At a Glance

| Dataset Attribute | Value / Description |
| --- | --- |
| **Primary Source** | IBM Telco Customer Churn (via Kaggle) |
| **Dataset Size** | 7,043 Customer Records $\times$ 21 Features |
| **Target Label** | `Churn` (`No`: 5,174 / 73.4% | `Yes`: 1,869 / 26.6%) |
| **Feature Types** | Categorical (Services, Contract), Numeric (`tenure`, `MonthlyCharges`), Binary |

---

## Project Workflow Architecture

```text
[ Data Ingestion & Hygiene ] ➡️ [ Systematic Validation & Cleaning ] ➡️ [ Encoding & Feature Scaling ]
                                                                                   │
[ Retention Strategy ] ⬅️ [ Model Evaluation (Recall/F1) ] ⬅️ [ Stratified Cross-Validation ]

```

---

## Tech Stack & Dependencies

* **Language:** Python 
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (Preprocessing, Model Selection, Metrics)
