# Credit Risk / Loan Default Prediction

A machine learning project that predicts the probability of credit card default using customer demographic information, repayment history, billing amounts, and payment behavior.

The project compares Logistic Regression, Random Forest, and XGBoost models and provides an interactive Streamlit application for making individual predictions.

> **Note:** This is an educational machine-learning project. It is not intended to be used as a real-world lending or credit approval system.

---

## 1. Project Overview

Credit default prediction is a classification problem where the goal is to identify customers who may fail to make their future credit card payments.

This project focuses on:

* Exploratory data analysis
* Feature engineering from repayment and payment behavior
* Handling an imbalanced target variable
* Comparing multiple machine learning models
* Evaluating precision, recall, F1, ROC-AUC, and PR-AUC
* Threshold analysis for higher recall
* Model explainability
* Building a Streamlit prediction application

---

## 2. Dataset

The project uses the **Default of Credit Card Clients** dataset from the UCI Machine Learning Repository.

Dataset size:

* **30,000 customers**
* **24 predictor features** after removing the customer ID
* **1 target variable**
* Target: `default`

Original target column:

```text
default.payment.next.month
```

Renamed in this project to:

```text
default
```

The dataset contains information about:

* Credit limit
* Sex
* Education
* Marriage status
* Age
* Repayment status for the previous six months
* Bill amounts for the previous six months
* Payment amounts for the previous six months

---

## 3. Target Distribution

The dataset is imbalanced.

| Class          | Customers | Percentage |
| -------------- | --------: | ---------: |
| 0 - No Default |    23,364 |     77.88% |
| 1 - Default    |     6,636 |     22.12% |

Because of this imbalance, accuracy alone is not sufficient to evaluate the model.

The project therefore also considers:

* Precision
* Recall
* F1 Score
* ROC-AUC
* PR-AUC

---

## 4. Exploratory Data Analysis

Initial data checks found:

* Dataset shape: `30,000 × 25`
* No missing values
* No duplicate rows
* No duplicate customer IDs
* Age range: 21–79
* Credit limit range: 10,000–1,000,000

### Repayment behavior

Repayment status was strongly associated with default in the dataset.

For example, the observed default rate for `PAY_0` increased substantially for customers with higher positive repayment-status values.

This indicates that recent repayment behavior is an important predictive signal.

Education categories showed differences in observed default rates as well, although several categories contained very few observations and should therefore not be overinterpreted.

---

## 5. Feature Engineering

In addition to the original variables, several behavioral features were created.

### Average repayment status

```python
df["avg_pay_status"] = df[pay_cols].mean(axis=1)
```

This summarizes repayment behavior across six months.

### Number of delayed payments

```python
df["num_delays"] = (df[pay_cols] >= 2).sum(axis=1)
```

This counts months with repayment-status values indicating a significant delay.

### Average bill amount

```python
df["avg_bill_amount"] = df[bill_cols].mean(axis=1)
```

### Average payment amount

```python
df["avg_payment_amount"] = df[payment_cols].mean(axis=1)
```

### Credit utilization

```python
df["credit_utilization"] = (
    df["avg_bill_amount"] / df["LIMIT_BAL"]
)
```

This represents the average bill relative to the customer's credit limit.

### Payment ratio

```python
df["payment_ratio"] = (
    df["avg_payment_amount"] /
    (df["avg_bill_amount"].abs() + 1)
)
```

This compares average payments with average bill amounts.

The original repayment, billing, and payment variables were retained along with these engineered features.

---

## 6. Models

Three classification models were evaluated.

### Logistic Regression

Used as a baseline model.

The features were standardized using `StandardScaler`.

### Random Forest

A tree-based ensemble model using:

```text
n_estimators = 200
max_depth = 10
min_samples_leaf = 5
```

### XGBoost

The final model uses:

```text
n_estimators = 300
max_depth = 4
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
```

---

## 7. Model Comparison

Results on the held-out test set using the default classification threshold of `0.50`:

| Metric    | Logistic Regression | Random Forest |    XGBoost |
| --------- | ------------------: | ------------: | ---------: |
| Accuracy  |              80.85% |        81.65% | **82.03%** |
| Precision |              65.29% |        66.01% | **67.08%** |
| Recall    |              28.64% |        35.12% | **36.85%** |
| F1 Score  |              39.81% |        45.84% | **47.57%** |
| ROC-AUC   |               0.747 |         0.779 |  **0.781** |
| PR-AUC    |               0.520 |         0.558 |  **0.564** |

XGBoost produced the highest values among the three tested models for each of these metrics on this evaluation split.

---

## 8. Threshold Analysis

The default classification threshold of `0.50` produced relatively low recall.

Because identifying more potential defaulters can be important in a risk-screening scenario, several lower thresholds were evaluated.

| Threshold | Precision |    Recall |        F1 |
| --------: | --------: | --------: | --------: |
|      0.20 |     43.1% |     66.9% |     52.5% |
|      0.25 | **49.2%** | **60.3%** | **54.2%** |
|      0.30 |     53.4% |     55.0% |     54.2% |
|      0.35 |     57.4% |     50.1% |     53.5% |
|      0.40 |     60.9% |     44.5% |     51.4% |
|      0.45 |     64.2% |     40.4% |     49.6% |
|      0.50 |     67.1% |     36.9% |     47.6% |

A threshold of **0.25** was selected as the project's high-recall operating point.

At this threshold:

```text
Precision = 49.17%
Recall    = 60.29%
F1 Score  = 54.16%
```

Confusion matrix:

```text
[[3846  827]
 [ 527  800]]
```

Therefore:

```text
True Negatives  = 3846
False Positives = 827
False Negatives = 527
True Positives  = 800
```

The threshold is a project-specific operating choice and should not be interpreted as a universal banking threshold.

---

## 9. Feature Importance

The XGBoost model placed substantial feature importance on repayment behavior.

Top features included:

| Feature              | Importance |
| -------------------- | ---------: |
| `num_delays`         |     0.3120 |
| `PAY_0`              |     0.2935 |
| `avg_pay_status`     |     0.0376 |
| `avg_bill_amount`    |     0.0215 |
| `avg_payment_amount` |     0.0192 |
| `PAY_2`              |     0.0182 |
| `payment_ratio`      |     0.0167 |
| `LIMIT_BAL`          |     0.0166 |
| `PAY_AMT2`           |     0.0165 |
| `credit_utilization` |     0.0163 |

`num_delays` and `PAY_0` are particularly prominent in the trained model.

Because `num_delays` is derived from repayment-status variables, these features contain related information and their importance should not be interpreted as independent causal effects.

---

## 10. Streamlit Application

The project includes an interactive Streamlit application.

The application allows a user to enter:

* Customer demographics
* Credit limit
* Six months of repayment history
* Six months of bill amounts
* Six months of payment amounts

The application then:

1. Calculates the engineered features.
2. Loads the trained XGBoost model.
3. Calculates default probability.
4. Applies the project threshold of `0.25`.
5. Displays the prediction.
6. Displays a project-defined risk category.
7. Shows customer risk indicators.
8. Displays individual feature contributions from XGBoost.

Example output:

```text
Default Probability: 63.4%

Prediction:
Potential Default

Risk Category:
High Risk
```

The exact probability depends on the input values.

---

## 11. Project Structure

```text
credit-risk-prediction/
│
├── data/
│   ├── default_of_credit_card_clients.xls
│   └── credit_risk_engineered.csv
│
├── notebooks/
│   └── 01_credit_risk_analysis.ipynb
│   
│
├── app/
│   └── streamlit_app.py
│
├── models/
│   ├── xgb_model.pkl
│   ├── feature_names.pkl
│   └── threshold.txt
├── screenshots/
|     └── streamlit_app.png
|
├── README.md
└── requirements.txt
```

---

## 12. Installation

Clone the repository and install the required packages:

```bash
pip install -r requirements.txt
```

Main dependencies include:

```text
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
openpyxl
joblib
streamlit
```

---

## 13. Running the Streamlit App

From the project root:

```bash
streamlit run app/streamlit_app.py
```

The application will open in a browser.

---

## 14. Model Files

The trained model and supporting files are stored in the `models/` directory.

### `xgb_model.pkl`

Saved XGBoost classification model.

### `feature_names.pkl`

Stores the exact feature order used during model training.

This ensures that the Streamlit application passes features to the model in the correct order.

### `threshold.txt`

Stores the classification threshold:

```text
0.25
```

---

## 15. Limitations

This project has several limitations:

* The dataset represents historical credit-card clients and may not represent every lending population.
* The model identifies statistical patterns rather than causal relationships.
* The selected threshold is specific to this project and its evaluation setup.
* A high-recall threshold increases false positives.
* Model predictions should not be treated as definitive financial decisions.
* Feature importance indicates model behavior, not causal importance.
* Further validation on external or more recent data would be required for real-world deployment.

---

## 16. Future Improvements

Possible improvements include:

* Hyperparameter tuning using cross-validation
* More systematic threshold optimization
* Calibration of predicted probabilities
* Cross-validation and confidence intervals
* SHAP-based explainability
* More detailed error analysis
* Fairness analysis across demographic groups
* Model monitoring after deployment
* Deployment using Streamlit Cloud or another hosting platform

---

## 17. Key Takeaways

The project demonstrates a complete machine-learning workflow:

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Engineering
     ↓
Train/Test Split
     ↓
Model Training
     ↓
Model Comparison
     ↓
Threshold Analysis
     ↓
XGBoost Model
     ↓
Streamlit Application
```

The final application converts the trained model into an interactive credit-risk prediction tool while making the prediction process easier to understand through probability, risk indicators, and feature contributions.

---

## 18. Dataset Reference

**UCI Machine Learning Repository — Default of Credit Card Clients Dataset**

https://archive.ics.uci.edu/dataset/350/default%2Bof%2B

---

## License

This project is intended for educational and portfolio purposes.
