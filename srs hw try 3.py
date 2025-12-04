import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

df = pd.read_csv("/Users/torcohen/Downloads/diamonds.csv")
x = df[['carat', 'y', 'x', 'z']]
y = df['price']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state=42)

model = LinearRegression()

model.fit(x_train, y_train)
print("Linear Regression:")
print(model.intercept_)
print(model.coef_)

y_pred = model.predict(x_test)

predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(predictions.head())

r2 = r2_score(y_test, y_pred)
print(f"R-squared score: {r2}")

loo = LeaveOneOut()
loo_errors = []

for train_idx, val_idx in loo.split(x):
    x_train_loo, x_val_loo = x.iloc[train_idx], x.iloc[val_idx]
    y_train_loo, y_val_loo = y.iloc[train_idx], y.iloc[val_idx]

    model_loo = LinearRegression()
    model_loo.fit(x_train_loo, y_train_loo)

    pred = model_loo.predict(x_val_loo)
    loo_errors.append((y_val_loo.values[0] - pred[0]) ** 2)

loo_mse = np.mean(loo_errors)
loo_rmse = np.sqrt(loo_mse)

print(f"LOOCV MSE: {loo_mse:.4f}")
print(f"LOOCV RMSE: {loo_rmse:.4f}")

kf = KFold(n_splits= 10, shuffle=True, random_state=42)

kf = LeaveOneOut()
kf_errors = []

for train_idx, val_idx in kf.split(x):
    x_train_kf, x_val_kf = x.iloc[train_idx], x.iloc[val_idx]
    y_train_kf, y_val_kf = y.iloc[train_idx], y.iloc[val_idx]

    model_kf = LinearRegression()
    model_kf.fit(x_train_kf, y_train_kf)

    pred = model_kf.predict(x_val_kf)
    kf_errors.append(mean_squared_error(y_val_kf, pred))

kf_mse = np.mean(kf_errors)
kf_rmse = np.sqrt(kf_mse)

print(f"LOOCV MSE: {kf_mse:.4f}")
print(f"LOOCV RMSE: {kf_rmse:.4f}")
