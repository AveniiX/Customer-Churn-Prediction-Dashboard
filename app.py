import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_model():
    with open('model/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/feature_names.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, features

@st.cache_data
def load_data():
    df = pd.read_csv('Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(
        df['TotalCharges'].median())
    df['Churn_Binary'] = (df['Churn'] == 'Yes').astype(int)
    return df

model, feature_names = load_model()
df = load_data()

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("Analyze customer churn patterns and predict churn risk.")

col1, col2, col3, col4 = st.columns(4)
churn_rate = df['Churn_Binary'].mean() * 100
col1.metric("Total Customers", f"{len(df):,}")
col2.metric("Churn Rate", f"{churn_rate:.1f}%")
col3.metric("Avg Monthly Charges", f"${df['MonthlyCharges'].mean():.0f}")
col4.metric("Avg Tenure", f"{df['tenure'].mean():.0f} months")

st.markdown("---")
tab1, tab2 = st.tabs(["EDA", "Predict Churn"])

with tab1:
    st.subheader("Churn Analysis")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x='tenure', color='Churn',
                           title='Tenure vs Churn',
                           barmode='overlay', opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(df, x='Churn', y='MonthlyCharges',
                     title='Monthly Charges vs Churn',
                     color='Churn')
        st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(df, x='Contract', color='Churn',
                       title='Contract Type vs Churn',
                       barmode='group')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Predict churn risk for a customer")

    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly = st.slider("Monthly Charges ($)", 18, 119, 65)
        total = st.number_input("Total Charges ($)",
                                 min_value=0.0, value=800.0)

    with col2:
        contract = st.selectbox("Contract Type",
            ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service",
            ["DSL", "Fiber optic", "No"])
        payment = st.selectbox("Payment Method",
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)",
             "Credit card (automatic)"])

    with col3:
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        paperless = st.selectbox("Paperless Billing", ["No", "Yes"])

    if st.button("Predict Churn Risk", type="primary"):
        input_data = {f: 0 for f in feature_names}
        input_data['tenure'] = tenure
        input_data['MonthlyCharges'] = monthly
        input_data['TotalCharges'] = total
        input_data['SeniorCitizen'] = 1 if senior == "Yes" else 0
        if 'Partner_Yes' in input_data:
            input_data['Partner_Yes'] = 1 if partner=="Yes" else 0
        if 'Dependents_Yes' in input_data:
            input_data['Dependents_Yes'] = 1 if dependents=="Yes" else 0
        if 'PaperlessBilling_Yes' in input_data:
            input_data['PaperlessBilling_Yes'] = 1 if paperless=="Yes" else 0
        contract_map = {
            "One year": "Contract_One year",
            "Two year": "Contract_Two year"
        }
        if contract in contract_map and contract_map[contract] in input_data:
            input_data[contract_map[contract]] = 1
        internet_map = {
            "Fiber optic": "InternetService_Fiber optic",
            "No": "InternetService_No"
        }
        if internet in internet_map and internet_map[internet] in input_data:
            input_data[internet_map[internet]] = 1

        X_input = pd.DataFrame([input_data])
        prob = model.predict_proba(X_input)[0][1]

        st.markdown("---")
        if prob >= 0.7:
            st.error(f"High Churn Risk: {prob*100:.1f}%")
            st.warning("Recommendation: Offer a loyalty discount or upgrade.")
        elif prob >= 0.4:
            st.warning(f"Medium Churn Risk: {prob*100:.1f}%")
            st.info("Recommendation: Send a satisfaction survey.")
        else:
            st.success(f"Low Churn Risk: {prob*100:.1f}%")
            st.info("This customer is likely to stay.")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob * 100, 1),
            title={'text': "Churn Risk %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "#EAF3DE"},
                    {'range': [40, 70], 'color': "#FAEEDA"},
                    {'range': [70, 100], 'color': "#FCEBEB"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Built with Scikit-learn + Streamlit + Plotly | Telco Churn Dataset")