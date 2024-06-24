import pandas as pd 

melbourne_data = pd.read_csv("./csv/melb_data.csv")

ds = melbourne_data.describe()
print(ds)
print(round(melbourne_data["Price"].mean()))