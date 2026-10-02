import pandas as pd

df = pd.read_csv("data/spam.csv", encoding="latin-1")

df = df[["v1", "v2"]]
df.columns = ["label", "message"]

print(df.head())
print(df.columns)

print(df["label"].value_counts())

print("Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()
print("Rows after removing duplicates:", len(df))


df["label"] = df["label"].map({"ham": 0, "spam": 1})

print(df.head())

df.to_csv("data/cleaned_spam.csv", index=False)