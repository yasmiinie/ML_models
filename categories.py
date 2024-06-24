from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder


#--------------------------------- Functions -------------------------------------- 
def score_dataset(X_train , X_val , y_train ,y_val):
    model = RandomForestRegressor(n_estimators=100, random_state = 0)
    model.fit( X_train , y_train)
    pred = model.predict(X_val)
    return mean_absolute_error(y_val , pred)




data = pd.read_csv("./csv/melb_data.csv")
y = data.Price
X = data.drop(["Price"] , axis = 1)

X_train , X_val ,y_train ,y_val = train_test_split(X , y , train_size = 0.8 , 
                                                   test_size = 0.2 , random_state = 0)


#cols with missing values
cols_missing = [col for col in X_train.columns 
                        if X_train[col].isnull().any() ]


X_train.drop(cols_missing , axis = 1 )
X_val.drop(cols_missing , axis = 1 )

# Search for categorical values 
cols_categ = [cols for cols in X_train.columns 
                if X_train[cols].nunique() < 10 and X_train[cols].dtype == "object" ]
print(cols_categ)
cols_num = [cols for cols in X_train
                    if X_train[cols].dtype in ['int64' , 'float64'] ]

mycols = cols_categ + cols_num
# On ne garde que les colonnes des num + colonnes des categories 
X_train = X_train[mycols]
X_val = X_val[mycols]

#---------------------------Methode 1 : drop the whole column---------------------------------


X_train_1 = X_train.select_dtypes(exclude = ["object"])
X_val_1 = X_val.select_dtypes(exclude = ["object"])


print("MAE 1" , score_dataset(X_train_1 , X_val_1 , y_train , y_val ))

#---------------------------Methode 2 : Encoding ---------------------------------
 
ord = OrdinalEncoder()

X_train_2 = X_train.copy()
X_val_2 = X_val.copy()

X_train_2[cols_categ] = ord.fit_transform(X_train[cols_categ])
X_val_2[cols_categ]= ord.transform(X_val[cols_categ])

print("MAE 2" , score_dataset(X_train_2 , X_val_2 , y_train , y_val ))

#---------------------------Methode 3 : Encoding ---------------------------------
OH = OneHotEncoder()

OH_X_train = pd.DataFrame(OH.fit_transform(X_train[cols_categ]))
OH_X_val = pd.DataFrame(OH.transform(X_val[cols_categ]))
#print(OH_X_train)
#print("index ")
# OH removes the indexes
OH_X_train.index = X_train.index
OH_X_val.index = X_val.index
#print(OH_X_train)
num_X_train = X_train.drop(cols_categ ,axis = 1)
num_X_val = X_val.drop(cols_categ ,axis = 1)

X_train_3 = pd.concat([num_X_train , OH_X_train] , axis = 1)
X_val_3 = pd.concat([num_X_val , OH_X_val] , axis = 1)

# Ensure all columns have string type
X_train_3.columns = X_train_3.columns.astype("string")
X_val_3.columns = X_val_3.columns.astype("string")

print(X_train_3)
print("MAE 2" , score_dataset(X_train_3 , X_val_3 , y_train , y_val ))
