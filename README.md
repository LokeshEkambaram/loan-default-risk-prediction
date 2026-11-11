# Loan Default Risk Prediction

Binary classification project that predicts whether a credit card client will default on their next payment, built with Python and scikit-learn.

## Dataset
UCI *Default of Credit Card Clients*, loaded through OpenML (data id 42477).
- Records: **30,000** (23 features, 22.1% default rate, no missing values)
- Target: default on next month's payment (1 = default)

## Approach
1. Stratified 80/20 train-test split (`random_state=42`)
2. Preprocessing pipeline: median imputation and scaling for numeric features, one-hot encoding for categorical features
3. Models compared with 5-fold stratified cross-validation: Logistic Regression, Random Forest, Histogram Gradient Boosting
4. Evaluation on the held-out test set: ROC-AUC, PR-AUC, recall, F1

## Results
| Model | CV ROC-AUC | Test ROC-AUC | PR-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|---|
| Logistic Regression (L2) | 0.727 | 0.708 | 0.490 | 0.620 | 0.367 | 0.461 |
| Hist Gradient Boosting | 0.782 | 0.775 | 0.549 | 0.618 | 0.465 | 0.530 |
| Random Forest | 0.777 | 0.774 | 0.549 | 0.563 | 0.525 | 0.543 |

Hist Gradient Boosting gave the best ROC-AUC. Recall and precision are at the default 0.5 threshold, with class weights used for imbalance.

Top default drivers (Random Forest importance): x6, x7, x1, x8, x9 (recent repayment-status features and credit limit).

## Limitations and next steps
- Classes are imbalanced (22% default), so accuracy is not used and class weights are applied; recall depends on the decision threshold, which a lender would tune to its cost of missing a defaulter.
- Next steps: threshold tuning, hyperparameter search, SHAP-based explanations, EDA visualisations.

## How to run
```bash
pip install -r requirements.txt
python loan_default_prediction.py
```

## Author
Lokesh Ekambaram - [LinkedIn](https://www.linkedin.com/in/lokesh-e-3233492a5/)
