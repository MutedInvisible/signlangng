import pandas as pd

# Load raw CSV
df = pd.read_csv("sign_data.csv", header=None)

# Drop missing rows
df.dropna(inplace=True)

# Convert all columns except last to float
for col in df.columns[:-1]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop any rows that couldn't convert
df.dropna(inplace=True)

# Save clean CSV
df.to_csv("sign_data_clean.csv", index=False, header=False)
print("Cleaned CSV saved as sign_data_clean.csv")
