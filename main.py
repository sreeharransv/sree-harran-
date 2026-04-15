import pandas as pd
from model import train_model

# Load dataset
data = pd.read_csv("dataset.csv")

# Features and labels
X = data[["hours_studied", "attendance"]]
y = data["passed"]

# Train model
model = train_model(X, y)

# Prediction
print("Enter hours studied:")
h = float(input())

print("Enter attendance:")
a = float(input())

prediction = model.predict([[h, a]])

if prediction[0] == 1:
    print("Result: PASS")
else:
    print("Result: FAIL")
