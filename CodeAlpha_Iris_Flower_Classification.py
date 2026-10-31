# CodeAlpha Task 1: Iris Flower Classification
# Dataset: built-in Scikit-learn Iris dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("First rows:\n", df.head())
print("\nShape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary:\n", df.describe())

sns.pairplot(df, hue="species")
plt.show()

plt.figure(figsize=(8, 5))
sns.heatmap(df.select_dtypes("number").corr(), annot=True, cmap="coolwarm")
plt.title("Iris Feature Correlation")
plt.tight_layout()
plt.show()

X = df.drop(columns="species")
y = df["species"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, predictions):.4f}")
print("\nClassification report:\n", classification_report(y_test, predictions))

cm = confusion_matrix(y_test, predictions, labels=model.classes_)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=X.columns)
print("Sample prediction:", model.predict(sample)[0])
