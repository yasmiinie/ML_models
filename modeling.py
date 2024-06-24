from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from sklearn.metrics import mean_absolute_error


data = pd.read_csv("./csv/melb_data.csv")
features = [ 'Rooms', 'Bathroom','Distance','Lattitude','Longtitude','Propertycount']
print(data.describe())
data = data.dropna(axis=0)
x = data[features]
y = data.Price

print(y.head())
# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(x, y)

print("Making predictions for the following 5 houses:")
print(x.head())
print("The predictions are")
prediction = melbourne_model.predict(x)

print(mean_absolute_error(y,prediction))