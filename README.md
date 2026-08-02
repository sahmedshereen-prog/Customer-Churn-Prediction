# Customer-Churn-Prediction

Predicting customer churn for a subscription-based service using classification models.

## Overview
- **Goal**: Predict whether a customer will churn (leave the service) or not.
- **Data**: Two CSV files — `customer_churn_dataset-training-master.csv` and `customer_churn_dataset-testing-master.csv`.
- **Columns**: CustomerID, Age, Gender, Tenure, Usage Frequency, Support Calls, Payment Delay, Subscription Type, Contract Length, Total Spend, Last Interaction, Churn (0 = stayed, 1 = churned).
- **Challenge**: Accuracy alone can be misleading, so the model is evaluated with precision and recall on the churn class as well.

## Approach
1. Load the training and testing files separately
2. Clean the data (drop ID column, handle missing values)
3. Encode categorical features (Gender, Subscription Type, Contract Length) — fit on train, applied to test
4. Scale numeric features — fit on train, applied to test
5. Handle class imbalance with **SMOTE** (applied only on the training set)
6. Train and compare multiple classifiers:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - SVM (RBF kernel)
7. Evaluate using precision, recall, F1-score, and ROC-AUC on the separate test set
8. Save the best-performing model

## How to run
```bash
pip install pandas numpy scikit-learn imbalanced-learn seaborn matplotlib joblib
```
1. Place `customer_churn_dataset-training-master.csv` and `customer_churn_dataset-testing-master.csv` in this folder.
2. Run:
```bash
python customer_churn_prediction.py
```

## Outputs
- `best_churn_model.pkl` — the best trained model
- `scaler.pkl` — the fitted feature scaler
- `model_comparison_results.csv` — metrics for all models
- `churn_distribution.png`, `roc_curve.png`, `precision_recall_curve.png` — visualizations

## Author
Uneeq Interns Task Submission
