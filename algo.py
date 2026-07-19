import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset.csv")

print("Source of Data: Kaggle - Amazon Product Reviews Dataset")

print("\nNumber of Records (Rows):", df.shape[0])
print("Number of Features (Columns):", df.shape[1])

print("\nFeatures / Attributes:")
for col in df.columns:
    print("-", col)

print("\nData Types:")
print(df.dtypes)


X = df["reviews.text"]
y = df["reviews.rating"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

print("\nTraining Percentage: {:.0f}%".format(len(X_train)/len(df)*100))
print("Testing Percentage: {:.0f}%".format(len(X_test)/len(df)*100))