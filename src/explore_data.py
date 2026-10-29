import pandas as pd

df = pd.read_csv("data/creditcard.csv")
print(df.isna().sum())
print(df.dtypes)
print(df.head(5))
print(df["Class"].value_counts(normalize=True) * 100)

X = df.drop(columns="Class")
y = df["Class"]
print(X.shape)
print(X.shape[1])
print(y.name)

