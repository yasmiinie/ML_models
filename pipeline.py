import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

#read data
data = pd.read_csv('./csv/melb_data.csv')

# Separate target from predictors
y = data.Price
X = data.drop(["Price"] , axis = 1)

# Divide data into training and validation subsets
X_train_full , X_val_full , y_train , y_val = train_test_split(X , y , train_size= 0.8 , test_size=0.2 , random_state = 0)

# Select num & cat cols
num_cols = [col for col in X_train_full.columns
                    if X_train_full[col].dtype in ['float64' , 'int64']]
cat_cols = [col for col in X_train_full.columns
                    if X_train_full[col].dtype == "object" and X[col].nunique() < 10]

#Keeping the num & col only
my_cols = num_cols + cat_cols
X_train = X_train_full[my_cols].copy()
X_val = X_val_full[my_cols].copy()

#PIPELINE
#Step 1: Define Preprocessing Steps
#We impute the num and we impute & one-hot-encode the cat
num_imputer = SimpleImputer(strategy =  "constant")
cat_imputer = Pipeline(steps= 
                       [
                           ('imputer', SimpleImputer(strategy = 'most_frequent')),
                           ("onehot" , OneHotEncoder(handle_unknown='ignore'))
                       ])

preprocessor = ColumnTransformer(transformers=
                        [
                            ("num" , num_imputer , num_cols)
                            ("cat" , cat_imputer , cat_cols)
                        ])

#Step 2: Define the Model
model = RandomForestRegressor(random_state=0)

#Step 3: Create and Evaluate the Pipeline
my_pipeline = Pipeline(steps=
                       [
                           ("preprosse" , preprocessor )
                           ("model" , model)
                       ])

my_pipeline.fit(X_train , y_train)
