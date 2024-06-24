import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

#----------------Functions ----------------
def get_mae(depth ,train_x , val_x , train_y , val_y):
    model = DecisionTreeRegressor(max_depth=depth,random_state=1)
    model.fit(train_x,train_y)
    pred = model.predict(val_x)
    return(mean_absolute_error(val_y,pred))





data = pd.read_csv("./csv/melb_data.csv")
data = data.dropna()

features = ['Rooms','Bathroom','Distance','Lattitude','Longtitude','Propertycount']
y = data.Price
x = data[features]

#Mean absolute error in-sample

#model = DecisionTreeRegressor(random_state=1)
#model.fit(x,y)
#print(mean_absolute_error(y,model.predict(x)))



train_x ,val_X, train_y , val_y = train_test_split(x,y,random_state=0)
for depth in [2,5,10 ,50,60,70 ,80, 100,500 ,5000]:
    print(get_mae(depth,train_x , val_X , train_y , val_y))
    


