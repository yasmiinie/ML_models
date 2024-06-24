from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import pandas as pd


data = pd.read_csv("./csv/melb_data.csv")
data = data.dropna()

features = ['Rooms','Bathroom','Distance','Lattitude','Longtitude','Propertycount']
x = data[features]
y = data.Price

forest_model = RandomForestRegressor(random_state=0)

forest_model.fit(x,y)
pred = forest_model.predict(x)
print(mean_absolute_error(y, pred))