import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "Models" / "churn_model.pkl"

CATEGORY_MAPS = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "MultipleLines": {"No": 0, "No phone service": 1, "Yes": 2},
    "InternetService": {"DSL": 0, "Fiber optic": 1, "No": 2},
    "OnlineSecurity": {"No": 0, "No internet service": 1, "Yes": 2},
    "OnlineBackup": {"No": 0, "No internet service": 1, "Yes": 2},
    "DeviceProtection": {"No": 0, "No internet service": 1, "Yes": 2},
    "TechSupport": {"No": 0, "No internet service": 1, "Yes": 2},
    "StreamingTV": {"No": 0, "No internet service": 1, "Yes": 2},
    "StreamingMovies": {"No": 0, "No internet service": 1, "Yes": 2},
    "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2},
    "PaperlessBilling": {"No": 0, "Yes": 1},
    "PaymentMethod": {
        "Bank transfer (automatic)": 0,
        "Credit card (automatic)": 1,
        "Electronic check": 2,
        "Mailed check": 3,
    },
}

FEATURE_ORDER = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


@st.cache_resource
def load_model():
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


def encode(category, value):
    return CATEGORY_MAPS[category][value]


st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊")
st.title("Customer Churn Prediction")
st.caption("Estimate churn risk from customer account and service information.")

model = load_model()

with st.form("customer"):
    account_col, billing_col = st.columns(2)

    with account_col:
        gender = st.selectbox("Gender", list(CATEGORY_MAPS["gender"]))
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", list(CATEGORY_MAPS["Partner"]))
        dependents = st.selectbox("Dependents", list(CATEGORY_MAPS["Dependents"]))
        tenure = st.number_input("Tenure (months)", 0, 100, 12)
        contract = st.selectbox("Contract", list(CATEGORY_MAPS["Contract"]))
        payment = st.selectbox(
            "Payment method", list(CATEGORY_MAPS["PaymentMethod"])
        )
        paperless = st.selectbox(
            "Paperless billing", list(CATEGORY_MAPS["PaperlessBilling"])
        )

    with billing_col:
        monthly = st.number_input("Monthly charges", 0.0, 200.0, 70.0)
        total = st.number_input(
            "Total charges", 0.0, 20000.0, float(monthly * tenure)
        )
        phone = st.selectbox(
            "Phone service", list(CATEGORY_MAPS["PhoneService"])
        )
        multiple_lines = st.selectbox(
            "Multiple lines", list(CATEGORY_MAPS["MultipleLines"])
        )
        internet = st.selectbox(
            "Internet service", list(CATEGORY_MAPS["InternetService"])
        )

    st.subheader("Internet Add-ons")
    service_columns = st.columns(3)
    internet_features = {}
    for index, feature in enumerate(
        [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
    ):
        with service_columns[index % 3]:
            internet_features[feature] = st.selectbox(
                feature.replace("Online", "Online ").replace(
                    "Device", "Device "
                ).replace("Tech", "Tech ").replace("Streaming", "Streaming "),
                list(CATEGORY_MAPS[feature]),
                key=feature,
            )

    submitted = st.form_submit_button("Predict churn risk")

if submitted:
    values = {
        "gender": encode("gender", gender),
        "SeniorCitizen": int(senior == "Yes"),
        "Partner": encode("Partner", partner),
        "Dependents": encode("Dependents", dependents),
        "tenure": tenure,
        "PhoneService": encode("PhoneService", phone),
        "MultipleLines": encode("MultipleLines", multiple_lines),
        "InternetService": encode("InternetService", internet),
        "Contract": encode("Contract", contract),
        "PaperlessBilling": encode("PaperlessBilling", paperless),
        "PaymentMethod": encode("PaymentMethod", payment),
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }
    values.update(
        {
            feature: encode(feature, value)
            for feature, value in internet_features.items()
        }
    )

    input_frame = pd.DataFrame(
        [[values[feature] for feature in FEATURE_ORDER]],
        columns=FEATURE_ORDER,
    )
    probability = float(model.predict_proba(input_frame)[0][1])

    st.metric("Estimated churn probability", f"{probability:.1%}")
    if probability >= 0.5:
        st.warning("Higher-risk account: prioritise a retention review.")
    else:
        st.success("Lower-risk account based on the current model.")

    st.caption(
        "Portfolio demonstration only. Predictions require human review and "
        "should not be used as the sole basis for customer decisions."
    )
