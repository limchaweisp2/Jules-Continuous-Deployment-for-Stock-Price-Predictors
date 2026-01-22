import pandas as pd
import json
import argparse
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def train_and_evaluate(data_path):
    # 1. Load data
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: {data_path} not found.")
        sys.exit(1)
    
    X = df.drop("target", axis=1)
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # 3. Predict
    y_pred = clf.predict(X_test)

    # 4. Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred)
    }

    # 5. Save metrics to json
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # 6. Generate and save confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix ({data_path})")
    plt.savefig("confusion_matrix.png")
    plt.close()

    print(f"Training complete using {data_path}. Metrics saved to metrics.json.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data_baseline.csv", help="Path to input data CSV")
    args = parser.parse_args()
    
    train_and_evaluate(args.data)
