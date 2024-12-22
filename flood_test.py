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

train.head()
test.head()
train.columns.tolist()
test.columns.tolist()

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
    df.loc[:, "mean_features"] = df[features].mean(axis=1)
    df.loc[:, "std_features"] = df[features].std(axis=1)
    df.loc[:, "max_features"] = df[features].max(axis=1)
    df.loc[:, "min_features"] = df[features].min(axis=1)
    df.loc[:, "range_features"] = df["max_features"] - df["min_features"]
    df.loc[:, "variance_features"] = df[features].var(axis=1)
    df.loc[:, "skewness_features"] = df[features].skew(axis=1)
    df.loc[:, "sum_features"] = df[features].sum(axis=1)
    df.loc[:, "kurtosis_features"] = df[features].kurtosis(axis=1)
    df.loc[:, "median_absolute_deviation"] = (df[features] - df[features].median(axis=1).values.reshape(-1, 1)).abs().mean(axis=1)
    return df

# Apply Feature Engineering
train_cleaned = feature_engineering(train_cleaned)
test_cleaned = feature_engineering(test)

# ---------------------------------
# Scaling Data
scaler = StandardScaler()
scaled_train = pd.DataFrame(scaler.fit_transform(train_cleaned), columns=train_cleaned.columns)
scaled_test = pd.DataFrame(scaler.transform(test_cleaned), columns=test_cleaned.columns)

#-------------------------------------------------------------------------
#catboost
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
y_pred_cb = cbr.predict(X_test)

from sklearn.metrics import r2_score as Rsquared
cb_rsquared = np.sqrt(Rsquared(y_test,y_pred_cb))
print("R2 for CatBoostRegressor:",np.mean(cb_rsquared))

end = time.time()
diff = end - start
print("Execution Time:",diff)

#plot catboost
import matplotlib.pyplot as plt
import numpy as np

# Calculate residuals
catboost_residuals = y_test - y_pred_cb

# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 10))

# Residual Plot
axes[0, 0].scatter(y_pred_cb, catboost_residuals, color='blue', alpha=0.5)
axes[0, 0].axhline(y=0, color='red', linestyle='--')
axes[0, 0].set_title('Residual Plot (CatBoost)')
axes[0, 0].set_xlabel('Predicted Values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].grid(True)

# Actual vs. Predicted Plot
axes[0, 1].scatter(y_test, y_pred_cb, color='blue', alpha=0.5)
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
#LightGBM 
import lightgbm
import numpy as np
import time
X = scaled_train
y=y

start = time.time()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

from sklearn.metrics import r2_score as Rsquared
lgbmr = lightgbm.LGBMRegressor()
lgbmr.fit(X_train, y_train)
y_pred_lgbm = lgbmr.predict(X_test)
lgbm_rsquared = np.sqrt(Rsquared(y_test, y_pred_lgbm))
print("R squared for LightGBM: ", lgbm_rsquared)

end = time.time()
diff = end - start
print('Execution Time: ',diff)

#plot lightgbm
from lightgbm import plot_importance
import matplotlib.pyplot as plt
import numpy as np

# Calculate residuals for LightGBM
lgbm_residuals = y_test - y_pred_lgbm

# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 10))

# Residual Plot
axes[0, 0].scatter(y_pred_lgbm, lgbm_residuals, color='blue', alpha=0.5)
axes[0, 0].axhline(y=0, color='red', linestyle='--')
axes[0, 0].set_title('Residual Plot (LGBMRegressor)')
axes[0, 0].set_xlabel('Predicted Values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].grid(True)

# Actual vs. Predicted Plot
axes[0, 1].scatter(y_test, y_pred_lgbm, color='blue', alpha=0.5)
axes[0, 1].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
axes[0, 1].set_title('Actual vs. Predicted Plot (LGBMRegressor)')
axes[0, 1].set_xlabel('Actual Values')
axes[0, 1].set_ylabel('Predicted Values')
axes[0, 1].grid(True)

# Feature Importance Plot
plot_importance(lgbmr, ax=axes[1, 0], importance_type='split')
axes[1, 0].set_title('Feature Importance (LGBMRegressor)')
axes[1, 0].set_xlabel('Feature Index')
axes[1, 0].set_ylabel('Importance')

# Residual Distribution Plot
axes[1, 1].hist(lgbm_residuals, bins=30, color='blue', alpha=0.5)
axes[1, 1].set_title('Residual Distribution (LGBMRegressor)')
axes[1, 1].set_xlabel('Residuals')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].grid(True)

# Styling
plt.tight_layout()
plt.show()

#---------------------------------------------------------------------
#XGBoost
import xgboost as xgb
from sklearn import preprocessing
X = scaled_train
y=y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

import time
start = time.time()
xgbr = xgb.XGBRegressor()

xgbr.fit(X_train,y_train)
y_pred_xgb = xgbr.predict(X_test)

from sklearn.metrics import r2_score as Rsquared
xgb_rmse = np.sqrt(Rsquared(y_test, y_pred_xgb))
print("R Squared for XGBoost: ",np.mean(xgb_rmse))

end = time.time()
diff = end - start
print('Execution Time: ',diff)

#-----------------------------------------------------------------------
#comparison
plt.figure(figsize=(10, 6))

# Histograms for LightGBM,XGB predictions
plt.hist(y_pred_lgbm, bins=50, alpha=0.5, color='green', label='LGBM Predictions')
plt.hist(y_pred_xgb, bins=50, alpha=0.5, color='red', label='XGB Predictions')

plt.xlabel('Predicted Flood Probability') 
plt.ylabel('Frequency')
plt.title('Distribution of Flood Probability Predictions: LightGBM vs. XGBoost')
plt.legend()
plt.show()

#-----------------------------------------------------------------------
#normality check for predictions
import scipy.stats as stats

r2_linear = lgbm_rsquared  # Adjusted to your LightGBM R-squared variable
r2_catboost = cb_rsquared  # Adjusted to your CatBoost R-squared variable
lgbm_test_predictions = y_pred_lgbm  # Predictions from LightGBM
catboost_test_predictions = y_pred_cb  # Predictions from CatBoost

# Create the Q-Q plots
plt.figure(figsize=(14, 6))

# LightGBM Q-Q plot
plt.subplot(1, 2, 1)
stats.probplot(lgbm_test_predictions, dist="norm", plot=plt)
plt.title('Q-Q Plot for LightGBM Predictions')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Ordered Values')
plt.grid(True)
plt.text(
    2, np.min(lgbm_test_predictions) + 0.2, 
    f'R-squared: {r2_linear:.4f}', 
    fontsize=12, color='red', 
    bbox=dict(facecolor='white', alpha=0.5)
)

# CatBoost Q-Q plot
plt.subplot(1, 2, 2)
stats.probplot(catboost_test_predictions, dist="norm", plot=plt)
plt.title('Q-Q Plot for CatBoost Predictions')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Ordered Values')
plt.grid(True)
plt.text(
    2, np.min(catboost_test_predictions) + 0.2, 
    f'R-squared: {r2_catboost:.4f}', 
    fontsize=12, color='red', 
    bbox=dict(facecolor='white', alpha=0.5)
)

# Adjust layout and display
plt.subplots_adjust(wspace=0.5) 
plt.show()
#-----------------------------------------------
#predicted values file
import pandas as pd

test_ids = range(len(y_pred_lgbm))

# Create the DataFrame
df_pred = pd.DataFrame({
    'ID': test_ids, 
    'FloodProbability': y_pred_lgbm * 0.7 + 
                        y_pred_cb * 0.0 + 
                        y_pred_xgb * 0.3
})

# Save to CSV
df_pred.to_csv('predictions.csv', index=False)

#hist for predicted FP
df_pred['FloodProbability'].hist(color='skyblue', bins=30, edgecolor='black')
plt.title('Predicted Flood Probability Distribution')
plt.xlabel('Flood Probability')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


