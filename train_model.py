from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "Models" / "churn_model.pkl"


def prepare_data():
    data = pd.read_csv(DATA_PATH)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data = data.dropna().drop(columns=["customerID"])
    data["Churn"] = data["Churn"].map({"No": 0, "Yes": 1})

    for column in data.select_dtypes(include="object").columns:
        data[column] = LabelEncoder().fit_transform(data[column])

    return data.drop(columns=["Churn"]), data["Churn"]


def main():
    features, target = prepare_data()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(classification_report(y_test, predictions))
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
