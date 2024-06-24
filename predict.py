import pandas as pd
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv("./csv/melb_data.csv")
data = data.dropna()
feature = ['Rooms','Lattitude','Longtitude','Propertycount']


x = data[feature]
y = data.Price

dataModel = DecisionTreeRegressor(random_state=1)
dataModel.fit(x,y)



print("prediction for :")
print(x.head())
print("Predictions")
print(dataModel.predict(x.head()))
