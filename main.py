import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_csv("train.csv")

# Simple cleaning
df = df.select_dtypes(include=['int64', 'float64'])
df = df.fillna(df.mean())

# Features & target
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Accuracy
score = model.score(X_test, y_test)
print("Model Score:", score)
