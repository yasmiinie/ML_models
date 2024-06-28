import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor


data = pd.read_csv("./csv/melb_data.csv")
 
y = data.Price
X = data.select_dtypes(exclude = ["object"])

X_train , X_val , y_train , y_val = train_test_split(X ,y)

model = XGBRegressor()
model.fit(X_train , y_train)
pred = model.predict(X_val)

print("MAE : " , mean_absolute_error(pred , y_val))