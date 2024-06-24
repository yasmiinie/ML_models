import pandas as pd
from sklearn.model_selection import train_test_split

home_data = pd.read_csv('./csv/train.csv')

print(home_data.columns.to_list())