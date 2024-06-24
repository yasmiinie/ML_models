from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def score_datasets(X_train , X_val ,y_train , y_val):
    model = RandomForestRegressor(random_state=0)
    model.fit(X_train,y_train)
    return (mean_absolute_error(y_val, model.predict(X_val)))


#------------------------------------------------------------------------

#Missing Values
data = pd.read_csv("./csv/melb_data.csv")

y = data.Price
predictors = data.drop(["Price"] , axis=1)
X = predictors.select_dtypes(exclude=["object"])

X_train , X_val , y_train , y_val = train_test_split(X,y,train_size=0.8, test_size=0.2,random_state=0)

#1st option : remove the whole column with a NAN value
print("-------------------1st option--------------------")
cols_with_miss = [ col for col in X_train.columns
                            if X_train[col].isnull().any()]
reduced_X_train = X_train.drop(cols_with_miss , axis=1)
reduced_X_val = X_val.drop(cols_with_miss ,axis = 1)

print("MAE 1= %d" , score_datasets(reduced_X_train,reduced_X_val, y_train ,y_val))


#2st option : replace the cell with mean or median
print("-------------------2nd option--------------------")
imputer = SimpleImputer()

imputed_X_train = pd.DataFrame(imputer.fit_transform(X_train))
imputed_X_val = pd.DataFrame(imputer.transform(X_val))

imputed_X_train.columns = X_train.columns
imputed_X_val.columns = X_val.columns

print("MAE 2 = %d" , score_datasets(imputed_X_train,imputed_X_val, y_train ,y_val))


# Approach 3 (An Extension to Imputation)
print("-------------------3rd option--------------------")

# Make copy to avoid changing original data (when imputing)
X_train_plus = X_train.copy()
X_val_plus = X_val.copy()


for col in cols_with_miss:
    X_train_plus [col +"_was_missing"] = X_train_plus[col].isnull()
    X_val_plus[col +"_was_missing"] = X_val_plus[col].isnull()
    
  

    
#imputation 
imputer2 = SimpleImputer()
imp_X_train = pd.DataFrame(imputer2.fit_transform(X_train_plus))
imp_X_val = pd.DataFrame(imputer2.transform(X_val_plus))

# Imputation removed column names; put them back
imp_X_train.columns = X_train_plus.columns
imp_X_val.columns = X_val_plus.columns


print("MAE 3 = %d" , score_datasets(imp_X_train,imp_X_val, y_train ,y_val))




# Shape of training data (num_rows, num_columns)
print(X_train.shape)

# Number of missing values in each column of training data
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column >0])