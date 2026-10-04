import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb


# -----------------------------
# Load model and project files
# -----------------------------
model = joblib.load("models/xgb_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")

with open("models/threshold.txt", "r") as f:
    threshold = float(f.read())


# -----------------------------
# Page settings
# -----------------------------
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Risk / Default Prediction")
st.write(
    "Enter customer information to estimate the probability "
    "of credit card default."
)

st.info(
    "This is an educational machine-learning project and "
    "should not be used as a real lending decision system."
)


# -----------------------------
# Customer information
# -----------------------------
st.subheader("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    limit_bal = st.number_input(
        "Credit Limit",
        min_value=10000,
        max_value=1000000,
        value=200000,
        step=10000
    )

with col2:
    sex = st.selectbox(
        "Sex",
        options=[1, 2],
        format_func=lambda x: "Male" if x == 1 else "Female"
    )

with col3:
    age = st.number_input(
        "Age",
        min_value=21,
        max_value=79,
        value=30
    )


col1, col2 = st.columns(2)

with col1:
    education = st.selectbox(
        "Education",
        options=[0, 1, 2, 3, 4, 5, 6],
        format_func=lambda x: {
            0: "0 - Unknown",
            1: "1 - Graduate School",
            2: "2 - University",
            3: "3 - High School",
            4: "4 - Other",
            5: "5 - Other",
            6: "6 - Other"
        }[x]
    )

with col2:
    marriage = st.selectbox(
        "Marriage",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "0 - Unknown",
            1: "1 - Married",
            2: "2 - Single",
            3: "3 - Other"
        }[x]
    )


# -----------------------------
# Repayment history
# -----------------------------
st.subheader("Repayment History")

st.write(
    "Values represent repayment status for recent months. "
    "Higher positive values generally indicate more serious delays."
)

pay_labels = [
    "PAY_0 - Most Recent Month",
    "PAY_2 - Previous Month",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6"
]

pay_values = []

cols = st.columns(3)

for i, label in enumerate(pay_labels):
    with cols[i % 3]:
        value = st.selectbox(
            label,
            options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
            index=2,
            key=f"pay_{i}"
        )
        pay_values.append(value)

pay_0, pay_2, pay_3, pay_4, pay_5, pay_6 = pay_values


# -----------------------------
# Bill amounts
# -----------------------------
st.subheader("Recent Bill Amounts")

bill_labels = [
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6"
]

bill_values = []

cols = st.columns(3)

for i, label in enumerate(bill_labels):
    with cols[i % 3]:
        value = st.number_input(
            label,
            min_value=-100000,
            max_value=2000000,
            value=50000,
            step=5000,
            key=f"bill_{i}"
        )
        bill_values.append(value)

bill_1, bill_2, bill_3, bill_4, bill_5, bill_6 = bill_values


# -----------------------------
# Payment amounts
# -----------------------------
st.subheader("Recent Payment Amounts")

payment_labels = [
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6"
]

payment_values = []

cols = st.columns(3)

for i, label in enumerate(payment_labels):
    with cols[i % 3]:
        value = st.number_input(
            label,
            min_value=0,
            max_value=2000000,
            value=10000,
            step=1000,
            key=f"payment_{i}"
        )
        payment_values.append(value)

pay_amt_1, pay_amt_2, pay_amt_3, pay_amt_4, pay_amt_5, pay_amt_6 = payment_values


# -----------------------------
# Prediction button
# -----------------------------
st.divider()

if st.button("🔍 Predict Credit Risk", use_container_width=True):

    # Original repayment features
    pay_history = [
        pay_0, pay_2, pay_3,
        pay_4, pay_5, pay_6
    ]

    # Original bill features
    bill_history = [
        bill_1, bill_2, bill_3,
        bill_4, bill_5, bill_6
    ]

    # Original payment features
    payment_history = [
        pay_amt_1, pay_amt_2, pay_amt_3,
        pay_amt_4, pay_amt_5, pay_amt_6
    ]

    # -------------------------
    # Engineered features
    # -------------------------
    avg_pay_status = np.mean(pay_history)

    num_delays = sum(
        1 for value in pay_history
        if value >= 2
    )

    avg_bill_amount = np.mean(bill_history)

    avg_payment_amount = np.mean(payment_history)

    credit_utilization = (
        avg_bill_amount / limit_bal
    )

    payment_ratio = (
        avg_payment_amount /
        (abs(avg_bill_amount) + 1)
    )

    # -------------------------
    # Create model input
    # -------------------------
    input_data = pd.DataFrame([{
        "LIMIT_BAL": limit_bal,
        "SEX": sex,
        "EDUCATION": education,
        "MARRIAGE": marriage,
        "AGE": age,

        "PAY_0": pay_0,
        "PAY_2": pay_2,
        "PAY_3": pay_3,
        "PAY_4": pay_4,
        "PAY_5": pay_5,
        "PAY_6": pay_6,

        "BILL_AMT1": bill_1,
        "BILL_AMT2": bill_2,
        "BILL_AMT3": bill_3,
        "BILL_AMT4": bill_4,
        "BILL_AMT5": bill_5,
        "BILL_AMT6": bill_6,

        "PAY_AMT1": pay_amt_1,
        "PAY_AMT2": pay_amt_2,
        "PAY_AMT3": pay_amt_3,
        "PAY_AMT4": pay_amt_4,
        "PAY_AMT5": pay_amt_5,
        "PAY_AMT6": pay_amt_6,

        "avg_pay_status": avg_pay_status,
        "num_delays": num_delays,
        "avg_bill_amount": avg_bill_amount,
        "avg_payment_amount": avg_payment_amount,
        "credit_utilization": credit_utilization,
        "payment_ratio": payment_ratio
    }])

    # Make sure feature order is exactly the same as training
    input_data = input_data[feature_names]

    # -------------------------
    # Prediction
    # -------------------------
    probability = model.predict_proba(input_data)[0][1]

    prediction = int(probability >= threshold)

    # -------------------------
    # Risk category
    # -------------------------
    if probability < 0.20:
        risk_category = "Low Risk"
    elif probability < 0.40:
        risk_category = "Moderate Risk"
    else:
        risk_category = "High Risk"

    # -------------------------
    # Display results
    # -------------------------
    st.subheader("Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Default Probability",
            f"{probability:.1%}"
        )

    with col2:
        if prediction == 1:
            st.error("Potential Default")
        else:
            st.success("Lower Default Risk")

    with col3:
        st.metric(
            "Risk Category",
            risk_category
        )

    st.progress(float(probability))

    st.caption(
        f"Classification threshold used by the model: {threshold:.2f}"
    )

    # -------------------------
    # Important customer factors
    # -------------------------
    st.subheader("Customer Risk Indicators")

    indicators = []

    if num_delays >= 2:
        indicators.append(
            f"⚠️ {num_delays} recent repayment periods show delayed payment status."
        )
    else:
        indicators.append(
            "✅ Few recent repayment delays."
        )

    if credit_utilization > 0.50:
        indicators.append(
            f"⚠️ Credit utilization is relatively high ({credit_utilization:.1%})."
        )
    else:
        indicators.append(
            f"✅ Credit utilization is relatively moderate ({credit_utilization:.1%})."
        )

    if payment_ratio < 0.20:
        indicators.append(
            "⚠️ Average payments are relatively low compared with average bills."
        )
    else:
        indicators.append(
            "✅ Payments are relatively stronger compared with average bills."
        )

    if pay_0 >= 2:
        indicators.append(
            "⚠️ The most recent repayment status indicates a significant delay."
        )

    for indicator in indicators:
        st.write(indicator)

    # -------------------------
    # Model contribution
    # -------------------------
    st.subheader("Model Contribution")

    try:
        dmatrix = xgb.DMatrix(
            input_data,
            feature_names=feature_names
        )

        contributions = model.get_booster().predict(
            dmatrix,
            pred_contribs=True
        )[0]

        contribution_df = pd.DataFrame({
            "Feature": feature_names,
            "Contribution": contributions[:-1]
        })

        contribution_df["Absolute Contribution"] = (
            contribution_df["Contribution"].abs()
        )

        contribution_df = contribution_df.sort_values(
            "Absolute Contribution",
            ascending=False
        ).head(8)

        contribution_df["Direction"] = np.where(
            contribution_df["Contribution"] > 0,
            "Higher default probability",
            "Lower default probability"
        )

        st.dataframe(
            contribution_df[
                ["Feature", "Contribution", "Direction"]
            ],
            use_container_width=True,
            hide_index=True
        )

    except Exception:
        st.write(
            "Individual feature contribution could not be displayed."
        )