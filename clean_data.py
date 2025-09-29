import pandas as pd


df = pd.read_csv("sign_data.csv", header=None)


df.dropna(inplace=True)


for col in df.columns[:-1]:
    df[col] = pd.to_numeric(df[col], errors='coerce')


df.dropna(inplace=True)

df.to_csv("sign_data_clean.csv", index=False, header=False)
print("Cleaned CSV saved as sign_data_clean.csv")
