import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, plot_importance

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("Train Shape:", train.shape)
print("Test Shape:", test.shape)

# Drop 'id' from both datasets for analysis
train_id = train['id']
test_id = test['id']
train.drop('id', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)

# Target Variable
y = train['FloodProbability']
train.drop('FloodProbability', axis=1, inplace=True)

# Check for Null Values
print("Train Null Values:\n", train.isnull().sum())
print("Test Null Values:\n", test.isnull().sum())

# ---------------------------------
# Outlier Removal using IQR Method
def remove_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_cleaned = df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]
    return df_cleaned

train_cleaned = remove_outliers(train)
print(f"Shape after outlier removal: {train_cleaned.shape}")

train.head()
train_cleaned.head()
# Re-align target variable after outlier removal
y = y.loc[train_cleaned.index]

# ---------------------------------
# Feature Engineering Function
def feature_engineering(df):
    features = df.columns.tolist()
    df["mean_features"] = df[features].mean(axis=1)
    df["std_features"] = df[features].std(axis=1)
    df["max_features"] = df[features].max(axis=1)
    df["min_features"] = df[features].min(axis=1)
    df["range_features"] = df["max_features"] - df["min_features"]
    df["variance_features"] = df[features].var(axis=1)
    df["skewness_features"] = df[features].skew(axis=1)
    df["sum_features"] = df[features].sum(axis=1)

    # Additional statistical measures
    df['kurtosis_features'] = df[features].kurtosis(axis=1)
    df['median_absolute_deviation'] = (df[features] - df[features].median(axis=1).values.reshape(-1, 1)).abs().mean(axis=1)
    return df

# Apply Feature Engineering
train_cleaned = feature_engineering(train_cleaned)
test_cleaned = feature_engineering(test)

# ---------------------------------
# Scaling Data
scaler = StandardScaler()
scaled_train = pd.DataFrame(scaler.fit_transform(train_cleaned), columns=train_cleaned.columns)
scaled_test = pd.DataFrame(scaler.transform(test_cleaned), columns=test_cleaned.columns)

# ---------------------------------
# Cross Validation with XGBRegressor
xgb_params = {
    'n_estimators': 600,
    'max_depth': 10,
    'learning_rate': 0.06,
    'random_state': 42,
    'early_stopping_rounds': 10
}

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
xgb_predictions = np.zeros(len(scaled_train))
xgb_true_labels = np.zeros(len(scaled_train))
xgb_test_predictions = np.zeros(len(scaled_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(scaled_train, y)):
    X_train, X_val = scaled_train.iloc[train_idx], scaled_train.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # XGBRegressor
    xgb_model = XGBRegressor(**xgb_params)
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100
    )
    
    xgb_fold_preds = xgb_model.predict(X_val)
    xgb_fold_test_preds = xgb_model.predict(scaled_test)
    
    xgb_predictions[val_idx] = xgb_fold_preds
    xgb_true_labels[val_idx] = y_val
    xgb_test_predictions += xgb_fold_test_preds / n_splits

overall_metric_xgb = r2_score(xgb_true_labels, xgb_predictions)
print("Overall R^2 (XGBRegressor):", overall_metric_xgb)

# ---------------------------------
# Visualization of Results
xgb_residuals = xgb_predictions - xgb_true_labels

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 10))

# Residual Plot
axes[0, 0].scatter(xgb_predictions, xgb_residuals, color='blue', alpha=0.5)
axes[0, 0].axhline(y=0, color='red', linestyle='--')
axes[0, 0].set_title('Residual Plot (XGBoost)')

# Actual vs Predicted
axes[0, 1].scatter(xgb_true_labels, xgb_predictions, color='blue', alpha=0.5)
axes[0, 1].plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--')
axes[0, 1].set_title('Actual vs Predicted (XGBoost)')

# Feature Importance
plot_importance(xgb_model, ax=axes[1, 0])
axes[1, 0].set_title('Feature Importance (XGBoost)')

# Residual Histogram
axes[1, 1].hist(xgb_residuals, bins=30, color='blue', alpha=0.5)
axes[1, 1].set_title('Residual Distribution (XGBoost)')

plt.tight_layout()
#plt.show()
#-------------------------------------------------------------------------
from catboost import CatBoostRegressor, Pool
#scaled_train.columns.tolist()
#scaled_test.columns.tolist()
X = scaled_train
y=y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)

pool_train = Pool(X_train,y_train)
pool_test = Pool(X_test)

import time
start = time.time()
cbr = CatBoostRegressor()
cbr.fit(pool_train)
y_pred = cbr.predict(X_test)

from sklearn.metrics import r2_score as Rsquared
cb_rsquared = np.sqrt(Rsquared(y_test,y_pred))
print("R2 for CatBoostRegressor:",np.mean(cb_rsquared))

end = time.time()
diff = end - start
print("Execution Time:",diff)

#----------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

# Calculate residuals
catboost_residuals = y_test - y_pred

# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 10))

# Residual Plot
axes[0, 0].scatter(y_pred, catboost_residuals, color='blue', alpha=0.5)
axes[0, 0].axhline(y=0, color='red', linestyle='--')
axes[0, 0].set_title('Residual Plot (CatBoost)')
axes[0, 0].set_xlabel('Predicted Values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].grid(True)

# Actual vs. Predicted Plot
axes[0, 1].scatter(y_test, y_pred, color='blue', alpha=0.5)
axes[0, 1].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
axes[0, 1].set_title('Actual vs. Predicted Plot (CatBoost)')
axes[0, 1].set_xlabel('Actual Values')
axes[0, 1].set_ylabel('Predicted Values')
axes[0, 1].grid(True)

# Feature Importance Plot
feature_importances = cbr.get_feature_importance(prettified=True)
axes[1, 0].bar(feature_importances['Feature Id'], feature_importances['Importances'], color='blue', alpha=0.7)
axes[1, 0].set_title('Feature Importance (CatBoost)')
axes[1, 0].set_xlabel('Feature')
axes[1, 0].set_ylabel('Importance')
axes[1, 0].tick_params(axis='x', rotation=90)  # Rotate x-axis labels for readability

# Residual Distribution Plot
axes[1, 1].hist(catboost_residuals, bins=30, color='blue', alpha=0.5)
axes[1, 1].set_title('Residual Distribution (CatBoost)')
axes[1, 1].set_xlabel('Residuals')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].grid(True)

# Add overall styling
plt.tight_layout()
plt.show()

#----------------------------------------------------------------