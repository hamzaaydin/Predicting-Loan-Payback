# Kaggle Loan Payback Prediction - AUC Improvement Suggestions

## 🎯 Current Status
- **Current AUC**: 0.9223 (very close to your goal!)
- **Current F1 Score**: 0.9431
- **Model**: CatBoost only

## 📊 High-Impact Suggestions

### 1. Feature Engineering ⭐⭐⭐
You're currently only using basic features. Add these engineered features:

```python
# After loading data, before encoding
# Create interaction features
train_df['income_to_loan_ratio'] = train_df['annual_income'] / (train_df['loan_amount'] + 1)
train_df['credit_score_to_dti'] = train_df['credit_score'] / (train_df['debt_to_income_ratio'] + 0.01)
train_df['loan_payment_burden'] = (train_df['loan_amount'] * train_df['interest_rate']) / (train_df['annual_income'] + 1)
train_df['disposable_income'] = train_df['annual_income'] * (1 - train_df['debt_to_income_ratio'])
train_df['risk_score'] = train_df['debt_to_income_ratio'] * train_df['interest_rate'] / (train_df['credit_score'] + 1)

# Apply same transformations to test
test_df['income_to_loan_ratio'] = test_df['annual_income'] / (test_df['loan_amount'] + 1)
test_df['credit_score_to_dti'] = test_df['credit_score'] / (test_df['debt_to_income_ratio'] + 0.01)
test_df['loan_payment_burden'] = (test_df['loan_amount'] * test_df['interest_rate']) / (test_df['annual_income'] + 1)
test_df['disposable_income'] = test_df['annual_income'] * (1 - test_df['debt_to_income_ratio'])
test_df['risk_score'] = test_df['debt_to_income_ratio'] * test_df['interest_rate'] / (test_df['credit_score'] + 1)
```

### 2. Model Ensembling ⭐⭐⭐
Combine multiple models for better performance:

```python
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
import numpy as np

# CatBoost
cat_model = CatBoostClassifier(
    iterations=3000,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=5,
    random_seed=42,
    eval_metric="AUC",
    verbose=200
)

# LightGBM
lgb_model = LGBMClassifier(
    n_estimators=3000,
    learning_rate=0.03,
    max_depth=8,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)

# XGBoost
xgb_model = XGBClassifier(
    n_estimators=3000,
    learning_rate=0.03,
    max_depth=8,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='auc',
    verbose=0
)

# Train all models
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
lgb_model.fit(X_train, y_train, eval_set=(X_val, y_val), callbacks=[lgb.early_stopping(100)])
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)

# Ensemble predictions (weighted average)
cat_pred = cat_model.predict_proba(X_val)[:, 1]
lgb_pred = lgb_model.predict_proba(X_val)[:, 1]
xgb_pred = xgb_model.predict_proba(X_val)[:, 1]

# Experiment with weights
ensemble_pred = 0.4 * cat_pred + 0.3 * lgb_pred + 0.3 * xgb_pred

print("Ensemble ROC AUC:", roc_auc_score(y_val, ensemble_pred))
```

### 3. Hyperparameter Tuning ⭐⭐
Your CatBoost parameters are good but can be optimized:

```python
from catboost import CatBoostClassifier

cat_model = CatBoostClassifier(
    iterations=5000,           # Increased from 2000
    learning_rate=0.02,        # Decreased from 0.05 for better convergence
    depth=7,                   # Try 6-9
    l2_leaf_reg=3,            # Try 1-7
    min_data_in_leaf=20,      # Add regularization
    bagging_temperature=1,     # Add randomness
    random_strength=1,         # Add randomness
    random_seed=42,
    eval_metric="AUC",
    early_stopping_rounds=200,  # Add early stopping
    verbose=200
)

cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
```

### 4. Cross-Validation Strategy ⭐⭐
Replace single train/val split with stratified K-fold:

```python
from sklearn.model_selection import StratifiedKFold
import numpy as np

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_predictions = np.zeros(len(X))
test_predictions = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n=== Fold {fold + 1} ===")
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    model = CatBoostClassifier(
        iterations=5000,
        learning_rate=0.02,
        depth=7,
        l2_leaf_reg=3,
        random_seed=42,
        eval_metric="AUC",
        verbose=False
    )
    
    model.fit(X_train_fold, y_train_fold, eval_set=(X_val_fold, y_val_fold), use_best_model=True)
    
    oof_predictions[val_idx] = model.predict_proba(X_val_fold)[:, 1]
    test_predictions += model.predict_proba(X_test)[:, 1] / n_folds

print(f"\nOverall OOF AUC: {roc_auc_score(y, oof_predictions)}")

# Use test_predictions for final submission
```

### 5. Handle Outliers Better ⭐
You identified outliers but didn't handle them:

```python
# Instead of removing, cap outliers (winsorization)
from scipy.stats import mstats

for col in num_cols:
    train_df[col] = mstats.winsorize(train_df[col], limits=[0.01, 0.01])
    test_df[col] = mstats.winsorize(test_df[col], limits=[0.01, 0.01])
```

### 6. Polynomial Features for Key Interactions ⭐
```python
from sklearn.preprocessing import PolynomialFeatures

# Apply to most important features only
key_features = ['credit_score', 'debt_to_income_ratio', 'annual_income']
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)

train_poly = poly.fit_transform(train_df[key_features])
test_poly = poly.transform(test_df[key_features])

# Add back to dataframes with proper column names
poly_cols = poly.get_feature_names_out(key_features)
train_df[poly_cols] = train_poly
test_df[poly_cols] = test_poly
```

### 7. Target Encoding for Categorical Features ⭐⭐
Instead of one-hot encoding, try target encoding:

```python
from category_encoders import TargetEncoder

cat_cols = ["gender", "marital_status", "education_level", "employment_status", "loan_purpose", "grade_subgrade"]

encoder = TargetEncoder(cols=cat_cols, smoothing=1.0)
train_df[cat_cols] = encoder.fit_transform(train_df[cat_cols], train_df['loan_paid_back'])
test_df[cat_cols] = encoder.transform(test_df[cat_cols])
```

## 🎓 Priority Implementation Order

1. **Feature Engineering** (Quickest win)
2. **Cross-Validation** (More robust evaluation)
3. **Hyperparameter Tuning** (Optimize single model first)
4. **Model Ensembling** (Combine best models)

## 📈 Expected Impact
- Feature Engineering: **+0.003-0.005** AUC
- Cross-Validation: Better generalization
- Better Hyperparameters: **+0.002-0.003** AUC
- Ensembling: **+0.005-0.010** AUC

**Total Expected:** AUC **0.930-0.935** 🎯

## 📝 Implementation Notes

- **Dependencies to install**: `lightgbm`, `xgboost`, `category_encoders`, `scipy`
- **Runtime**: Feature engineering and ensembling will take longer to train
- **Memory**: Ensembling with multiple models requires more RAM
- **Validation**: Always use cross-validation to avoid overfitting

## 🔍 Additional Tips

- **Feature Selection**: After adding features, use feature importance to select the best ones
- **Blending vs Stacking**: Try different ensemble methods
- **Post-processing**: Consider calibration of probabilities
- **Data Leakage**: Ensure no target information leaks into features during cross-validation

---
*Generated on November 2, 2025*