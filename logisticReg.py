import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt

csv_path = "ssh_anomaly_dataset.csv"
df = pd.read_csv(csv_path)

df["target"] = (df["label"] != "normal").astype(int)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute

df = df.drop(columns=["label", "timestamp", "detail"])

X = df.drop(columns=["target"])
y = df["target"]

categorical_cols = ["source_ip", "username", "event_type", "status"]
numeric_cols = ["hour", "minute"]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

clf = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", LogisticRegression(max_iter=1000, n_jobs=-1)),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print("Classification report")
print(classification_report(y_test, y_pred))

print("Confusion matrix (TN, FP / FN, TP)")
print(confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {roc_auc:.4f}")

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

print("\nPrzykładowe predykcje dla pierwszych 10 logów z testu")
example = X_test.head(10)
example_pred = clf.predict(example)
example_proba = clf.predict_proba(example)[:, 1]

for i, (idx, row) in enumerate(example.iterrows()):
    print(f"\nRekord {i} (index {idx}):")
    print(row.to_dict())
    print(f"  -> przewidywana klasa (0=normal, 1=anomalia): {int(example_pred[i])}")
    print(f"  -> prawdopodobieństwo anomalii: {example_proba[i]:.4f}")
