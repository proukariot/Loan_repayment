# Loan Default Prediction

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## Overview

This project builds a machine learning pipeline to predict whether a loan will be **fully repaid or defaulted** (`loan_status`). Starting from raw Lending Club data, the workflow covers data cleaning, exploratory data analysis (EDA), feature engineering, clustering, model training, dimensionality reduction with PCA, and hyperparameter tuning.

Six classifiers are benchmarked:

| Model | Type |
|---|---|
| BaggingClassifier | Ensemble (bagging) |
| XGBoost | Gradient boosting |
| Logistic Regression | Linear |
| AdaBoost | Ensemble (boosting) |
| Gradient Boosting | Ensemble (boosting) |
| Random Forest | Ensemble (bagging) |

The final model (Logistic Regression tuned with GridSearchCV + SMOTE for class imbalance) achieves a **ROC-AUC > 0.70** on the held-out test set.

---

## Repository Structure

```
Loan_repayment/
├── 1_Data_Cleaning.ipynb          # Drop irrelevant/leaky columns, handle nulls
├── 2_EDA.ipynb                    # Exploratory Data Analysis and visualisations
├── 3_Feature_Engineering.ipynb    # Encoding, outlier treatment, feature creation
├── 4_Modeling.ipynb               # Clustering, model training, PCA, tuning
├── Loan_data.csv                  # Raw dataset (source: Kaggle)
├── c_data.csv                     # Cleaned intermediate data
├── modeling_data.csv              # Feature-engineered data ready for modelling
├── requirements.txt               # Python dependencies
└── README.md
```

---

## Dataset

The raw dataset (`Loan_data.csv`) is sourced from **Kaggle** (LendingClub loan data). After cleaning, the following features are used for modelling:

| Feature | Description |
|---|---|
| `loan_amnt` | Requested loan amount |
| `credit_month` | Loan term in months (36 / 60) |
| `int_rate` | Interest rate |
| `emp_length` | Employment length (years) |
| `annual_inc` | Annual income |
| `verification_status` | Income verification status |
| `dti` | Debt-to-income ratio |
| `delinq_2yrs` | Delinquencies in last 2 years |
| `fico_range` | FICO credit score |
| `inq_last_6mths` | Credit inquiries in last 6 months |
| `open_acc` | Number of open credit lines |
| `revol_bal` | Revolving balance |
| `total_acc` | Total credit lines |
| `pub_rec_bankruptcies` | Public record bankruptcies |
| `grade_numeric` | Loan grade (encoded) |
| `years_of_credit_history` | Age of oldest credit line |
| `home_ownership_*` | One-hot encoded home ownership |
| `purpose_*` | One-hot encoded loan purpose |
| **`loan_status`** | **Target: 1 = Fully Paid, 0 = Charged Off** |

---

## Pipeline

### 1. Data Cleaning (`1_Data_Cleaning.ipynb`)
- Remove columns that are entirely null, have a single unique value, or contain >60 % missing values
- Drop leaky columns (post-origination payment data)
- Filter target to binary classes: *Fully Paid* and *Charged Off*

### 2. Exploratory Data Analysis (`2_EDA.ipynb`)
- FICO score vs. loan status distribution
- Credit age vs. default likelihood
- Correlation analysis across numeric features

### 3. Feature Engineering (`3_Feature_Engineering.ipynb`)
- Impute missing values in `emp_length` and `pub_rec_bankruptcies`
- Convert `grade`/`sub_grade` to a single numeric `grade_numeric`
- Derive `years_of_credit_history` from date columns
- Winsorise numerical outliers (IQR-based, 5 % tails)
- One-hot encode `home_ownership` and `purpose`

### 4. Modelling (`4_Modeling.ipynb`)
- **Clustering**: K-Means, Hierarchical (single & complete linkage), DBSCAN
- **Classification**: Train and compare six models using accuracy and ROC-AUC
- **PCA**: Reduce to 17 components retaining 95 % variance; re-evaluate all models
- **Hyperparameter tuning**: GridSearchCV with 10-fold cross-validation on Logistic Regression
- **Class imbalance**: SMOTE oversampling applied before final model training

---

## Results

| Model | Test Accuracy | ROC-AUC |
|---|---|---|
| BaggingClassifier | — | — |
| XGBoost | — | — |
| Logistic Regression | — | — |
| AdaBoost | — | — |
| Gradient Boosting | — | — |
| Random Forest | — | — |

> Run the notebooks end-to-end to populate the results table with values from your environment.

---

## Getting Started

### Prerequisites

- Python 3.9+
- Jupyter Lab or Jupyter Notebook

### Installation

```bash
git clone https://github.com/proukariot/Loan_repayment.git
cd Loan_repayment
pip install -r requirements.txt
```

### Running the Notebooks

Execute the notebooks **in order**:

```bash
jupyter lab
```

1. `1_Data_Cleaning.ipynb`
2. `2_EDA.ipynb`
3. `3_Feature_Engineering.ipynb`
4. `4_Modeling.ipynb`

Each notebook reads from and writes to the CSV files in the project root, so the sequence must be respected.

---

## Acknowledgements

- Dataset: [LendingClub Loan Data – Kaggle](https://www.kaggle.com/)
- Libraries: scikit-learn, XGBoost, imbalanced-learn, yellowbrick, pandas, seaborn
