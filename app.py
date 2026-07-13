import joblib
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)


# ---------------------------------------------------------
# LOAD TRAINED MODEL AND SCALER
# ---------------------------------------------------------
@st.cache_resource
def load_model_and_scaler():
    trained_model = joblib.load("Customer_Churn_Model.pkl")
    trained_scaler = joblib.load("Customer_Churn_Scaler.pkl")

    return trained_model, trained_scaler


try:
    model, scaler = load_model_and_scaler()

except FileNotFoundError as error:
    st.error(
        "The trained model or scaler file could not be found. "
        "Make sure Customer_Churn_Model.pkl and "
        "Customer_Churn_Scaler.pkl are in the same folder as app.py."
    )

    st.code(str(error))
    st.stop()

except Exception as error:
    st.error("The model or scaler could not be loaded.")
    st.code(str(error))
    st.stop()


# ---------------------------------------------------------
# APPLICATION TITLE
# ---------------------------------------------------------
st.title("🏦 Bank Customer Churn Prediction System")

st.write(
    """
    This application uses a trained Random Forest machine learning model
    to estimate the probability that a bank customer may leave the bank.

    Enter the customer's demographic, financial and banking information
    below, and then click **Predict Customer Churn**.
    """
)

st.divider()


# ---------------------------------------------------------
# CUSTOMER INPUT FORM
# ---------------------------------------------------------
st.subheader("Customer Details")

column_1, column_2 = st.columns(2)


# LEFT COLUMN
with column_1:

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=650,
        step=1
    )

    geography = st.selectbox(
        "Geography",
        options=["France", "Germany", "Spain"]
    )

    gender = st.selectbox(
        "Gender",
        options=["Female", "Male"]
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=40,
        step=1
    )

    tenure = st.number_input(
        "Tenure with Bank (Years)",
        min_value=0,
        max_value=10,
        value=5,
        step=1
    )


# RIGHT COLUMN
with column_2:

    balance = st.number_input(
        "Account Balance",
        min_value=0.0,
        value=75000.0,
        step=1000.0,
        format="%.2f"
    )

    number_of_products = st.selectbox(
        "Number of Bank Products",
        options=[1, 2, 3, 4]
    )

    has_credit_card = st.selectbox(
        "Has Credit Card?",
        options=["No", "Yes"]
    )

    is_active_member = st.selectbox(
        "Is Active Member?",
        options=["No", "Yes"]
    )

    estimated_salary = st.number_input(
        "Estimated Annual Salary",
        min_value=0.0,
        value=100000.0,
        step=1000.0,
        format="%.2f"
    )


# ---------------------------------------------------------
# CONVERT TEXT INPUTS INTO NUMERICAL VALUES
# ---------------------------------------------------------

# Gender encoding
gender_male = 1 if gender == "Male" else 0

# Geography one-hot encoding
geography_germany = 1 if geography == "Germany" else 0
geography_spain = 1 if geography == "Spain" else 0

# France is represented when both Germany and Spain are zero

# Binary variables
credit_card_value = 1 if has_credit_card == "Yes" else 0
active_member_value = 1 if is_active_member == "Yes" else 0


# ---------------------------------------------------------
# CREATE MODEL INPUT
# ---------------------------------------------------------
# The feature order must be exactly the same as the order
# used when the model was trained in Google Colab.

feature_columns = [
    "Year",
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Geography_Germany",
    "Geography_Spain",
    "Gender_Male"
]


input_data = pd.DataFrame(
    [
        {
            "Year": 2025,
            "CreditScore": credit_score,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": number_of_products,
            "HasCrCard": credit_card_value,
            "IsActiveMember": active_member_value,
            "EstimatedSalary": estimated_salary,
            "Geography_Germany": geography_germany,
            "Geography_Spain": geography_spain,
            "Gender_Male": gender_male
        }
    ],
    columns=feature_columns
)


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
st.write("")

if st.button(
    "Predict Customer Churn",
    type="primary",
    use_container_width=True
):

    try:
        # Apply the same feature scaling used during training
        scaled_input = scaler.transform(input_data)

        # Generate binary prediction
        prediction = model.predict(scaled_input)[0]

        # Generate churn probability
        churn_probability = model.predict_proba(
            scaled_input
        )[0][1]

        risk_percentage = churn_probability * 100


        # -------------------------------------------------
        # RISK CLASSIFICATION
        # -------------------------------------------------
        if churn_probability >= 0.70:

            risk_level = "High Risk"

            recommendation = (
                "Immediate retention action is recommended. "
                "Contact the customer, identify possible dissatisfaction, "
                "and provide a personalised retention offer, fee waiver, "
                "loyalty benefit or relationship-manager assistance."
            )

        elif churn_probability >= 0.30:

            risk_level = "Medium Risk"

            recommendation = (
                "The customer should be monitored closely. "
                "Provide personalised communication, relevant product offers, "
                "customer-service follow-up and engagement incentives."
            )

        else:

            risk_level = "Low Risk"

            recommendation = (
                "Continue regular customer engagement and service monitoring. "
                "No immediate retention intervention is required."
            )


        customer_status = (
            "Likely to Churn"
            if prediction == 1
            else "Likely to Stay"
        )


        # -------------------------------------------------
        # DISPLAY RESULTS
        # -------------------------------------------------
        st.divider()
        st.subheader("Prediction Result")

        result_1, result_2, result_3 = st.columns(3)

        with result_1:
            st.metric(
                label="Churn Probability",
                value=f"{risk_percentage:.2f}%"
            )

        with result_2:
            st.metric(
                label="Risk Level",
                value=risk_level
            )

        with result_3:
            st.metric(
                label="Predicted Status",
                value=customer_status
            )


        # Risk-based message
        if risk_level == "High Risk":

            st.error(
                f"🔴 High Risk Customer\n\n{recommendation}"
            )

        elif risk_level == "Medium Risk":

            st.warning(
                f"🟡 Medium Risk Customer\n\n{recommendation}"
            )

        else:

            st.success(
                f"🟢 Low Risk Customer\n\n{recommendation}"
            )


        # Probability progress bar
        st.subheader("Churn Risk Indicator")

        st.progress(
            min(max(float(churn_probability), 0.0), 1.0)
        )

        st.write(
            f"Estimated probability of customer churn: "
            f"**{risk_percentage:.2f}%**"
        )


        # Show entered customer details
        with st.expander("View Customer Input Details"):

            display_data = pd.DataFrame(
                {
                    "Customer Attribute": [
                        "Credit Score",
                        "Geography",
                        "Gender",
                        "Age",
                        "Tenure",
                        "Account Balance",
                        "Number of Products",
                        "Credit Card Ownership",
                        "Active Member Status",
                        "Estimated Salary"
                    ],

                    "Entered Value": [
                        credit_score,
                        geography,
                        gender,
                        age,
                        tenure,
                        f"{balance:,.2f}",
                        number_of_products,
                        has_credit_card,
                        is_active_member,
                        f"{estimated_salary:,.2f}"
                    ]
                }
            )

            st.dataframe(
                display_data,
                use_container_width=True,
                hide_index=True
            )


    except ValueError as error:

        st.error(
            "Prediction could not be completed because the model input "
            "does not match the data structure used during training."
        )

        st.code(str(error))


    except Exception as error:

        st.error(
            "An unexpected error occurred while generating the prediction."
        )

        st.code(str(error))


# ---------------------------------------------------------
# MODEL INFORMATION
# ---------------------------------------------------------
st.divider()

with st.expander("About the Prediction Model"):

    st.write(
        """
        **Model used:** Random Forest Classifier

        **Model accuracy:** Approximately 86.3%

        **Prediction target:**

        - 0 = Customer Retained
        - 1 = Customer Churned

        **Risk classification:**

        - Low Risk: 0%–30%
        - Medium Risk: Above 30%–Below 70%
        - High Risk: 70% and above
        """
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.caption(
    "MBA Project: Predictive Modeling and Risk Scoring "
    "for Bank Customer Churn"
)