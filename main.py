import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# dataset
data = {
    "rainfall": [80, 60, 90, 30, 75, 50, 85, 40],
    "humidity": [85, 70, 88, 65, 80, 60, 90, 55],
    "cases": [40, 20, 45, 10, 35, 15, 50, 12],
    "risk": ["High", "Low", "High", "Low", "High", "Low", "High", "Low"]
}

df = pd.DataFrame(data)

# features & target
X = df[["rainfall", "humidity", "cases"]]
y = df["risk"]

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# prediction
predictions = model.predict(X_test)
print("Predictions:", predictions)

# accuracy
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# graph
plt.bar(["Accuracy"], [accuracy])
plt.title("Model Performance")
plt.show()