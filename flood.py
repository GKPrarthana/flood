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
#catboost
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
catboost_params = {
    'random_state':42,
    'learning_rate':0.0112770163043636001,
    'depth':8,
    'subsample':0.8675506657380021,
    'colsample_bylevel':0.7183884158632279,
    'min_data_in_leaf':98
}

catboost_predictions = np.zeros(len(scaled_train))
catboost_true_labels = np.zeros(len(scaled_train))
catboost_test_predictions = np.zeros(len(scaled_test))

for fold,(train_idx,val_idx) in enumerate(kf.split(scaled_train,y)):
    X_train, X_val = scaled_train.iloc[train_idx], scaled_train.iloc[val_idx]
    y_train,y_val = y.iloc[train_idx],y.iloc[val_idx]
    catboost_model = CatBoostRegressor(**catboost_params)
    catboost_model.fit(X_train,y_train, eval_set=(X_val,y_val), early_stopping_rounds=10)
    catboost_fold_preds = catboost_model.predict(scaled_test)
    catboost_fold_test_preds = catboost_model.predict(scaled_test)
    catboost_predictions[val_idx] = catboost_fold_preds
    catboost_true_labels[val_idx] = y_val
    catboost_test_predictions += catboost_fold_test_preds / n_splits

overall_metric_catboost = r2_score(catboost_true_labels, catboost_predictions)
print("Overall R^2 (CatBoostRegressor):", overall_metric_catboost)
#ValueError: shape mismatch: value array of shape (745305,) could not be broadcast to indexing result of shape (169820,) 
#plot
#---------------------------------------------------------------------------------------------------
catboost_residuals = catboost_predictions - catboost_true_labels
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 10))


axes[0, 0].scatter(catboost_predictions, catboost_residuals, color='blue', alpha=0.5)
axes[0, 0].axhline(y=0, color='red', linestyle='--')
axes[0, 0].set_title('Residual Plot (CatBoost)')
axes[0, 0].set_xlabel('Predicted Values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].grid(True)


axes[0, 1].scatter(catboost_true_labels, catboost_predictions, color='blue', alpha=0.5)
axes[0, 1].plot([min(catboost_true_labels), max(catboost_true_labels)], [min(catboost_true_labels), max(catboost_true_labels)], color='red', linestyle='--')
axes[0, 1].set_title('Actual vs. Predicted Plot (CatBoost)')
axes[0, 1].set_xlabel('Actual Values')
axes[0, 1].set_ylabel('Predicted Values')
axes[0, 1].grid(True)


catboost_model.get_feature_importance(prettified=True).plot(kind='bar', x='Feature Id', y='Importances', ax=axes[1, 0])
axes[1, 0].set_title('Feature Importance (CatBoost)')
axes[1, 0].set_xlabel('Feature')
axes[1, 0].set_ylabel('Importance')


axes[1, 1].hist(catboost_residuals, bins=30, color='blue', alpha=0.5)
axes[1, 1].set_title('Residual Distribution (CatBoost)')
axes[1, 1].set_xlabel('Residuals')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].grid(True)
plt.gcf().set_facecolor('cyan')
plt.tight_layout()
plt.show()