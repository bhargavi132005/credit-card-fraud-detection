import streamlit as st
import pandas as pd
import joblib
import os

# Configure the page to look wide and modern
st.set_page_config(
    page_title="SecureTx",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR DARK THEME & STYLING ---
st.markdown("""
<style>
    /* Main App Background */
    .stApp {
        background-color: #000000;
    }
    /* Sidebar Background */
    [data-testid="stSidebar"] {
        background-color: #0a0b0c;
        border-right: 1px solid #1f1f1f;
    }
    /* Text Colors */
    h1, h2, h3, p, label, .stMarkdown {
        color: #f0f2f6 !important;
    }
    /* Input field backgrounds */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #16181a !important;
        color: white !important;
    }
    /* Style the Scan Button */
    div.stButton > button, div.stFormSubmitButton > button {
        background-color: #ff3333;
        color: white;
        border-radius: 8px;
        height: 50px;
        font-weight: bold;
        font-size: 18px;
        border: 1px solid #ff3333;
        transition: all 0.3s ease;
        box-shadow: 0px 0px 10px rgba(255, 51, 51, 0.2);
    }
    /* Button Hover Effect */
    div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        background-color: #000000;
        color: #ff3333;
        box-shadow: 0px 0px 20px rgba(255, 51, 51, 0.6);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR: User Inputs ---
st.sidebar.header("📝 Transaction Details")
st.sidebar.markdown("Enter the transaction data below to scan for fraud.")

with st.sidebar.form("transaction_form"):
    # Transaction Type
    tx_type = st.selectbox("Transaction Type", ["CASH_OUT", "TRANSFER", "DEBIT", "PAYMENT", "CASH_IN"])

    # Monetary Amounts
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=150.00, step=10.0, format="%.2f", help="Click inside the box to type an exact amount.")
    oldbalance_orig = st.number_input("Sender Old Balance ($)", min_value=0.0, value=1000.00, step=100.0, format="%.2f", help="Click inside the box to type.")
    newbalance_orig = st.number_input("Sender New Balance ($)", min_value=0.0, value=850.00, step=100.0, format="%.2f", help="Click inside the box to type.")

    oldbalance_dest = st.number_input("Receiver Old Balance ($)", min_value=0.0, value=500.00, step=100.0, format="%.2f", help="Click inside the box to type.")
    newbalance_dest = st.number_input("Receiver New Balance ($)", min_value=0.0, value=650.00, step=100.0, format="%.2f", help="Click inside the box to type.")

    scan_button = st.form_submit_button("🔍 Scan Transaction", use_container_width=True, type="primary")


# --- MAIN DASHBOARD ---
st.title("🛡️ SecureTx: Real-Time Detection")
st.markdown("""
Welcome to the SecureTx monitoring dashboard. Enter transaction details in the sidebar and click **Scan Transaction** to evaluate the risk of fraud.
""")

st.divider()

if scan_button:
    with st.spinner("Analyzing transaction patterns..."):
        
        # --- 1. BUSINESS RULES ENGINE ---
        if amount > oldbalance_orig:
            st.error(f"🚫 **TRANSACTION DECLINED:** Insufficient funds. Attempted to send ${amount:,.2f} with only ${oldbalance_orig:,.2f} available.", icon="🚫")
            
        # --- 2. MACHINE LEARNING MODEL ---
        else:
            model_path = "results/xgboost_baseline.joblib"
        
            if os.path.exists(model_path):
                model = joblib.load(model_path)
            
                # 1. Format the inputs into a dictionary
                input_data = {
                    'amount': amount,
                    'oldbalanceOrg': oldbalance_orig,
                    'newbalanceOrig': newbalance_orig,
                    'oldbalanceDest': oldbalance_dest,
                    'newbalanceDest': newbalance_dest,
                    'balanceDiffOrig': oldbalance_orig - newbalance_orig,
                    'balanceDiffDest': newbalance_dest - oldbalance_dest,
                    'type_CASH_OUT': 1 if tx_type == 'CASH_OUT' else 0,
                    'type_DEBIT': 1 if tx_type == 'DEBIT' else 0,
                    'type_PAYMENT': 1 if tx_type == 'PAYMENT' else 0,
                    'type_TRANSFER': 1 if tx_type == 'TRANSFER' else 0,
                }
            
                input_df = pd.DataFrame([input_data])
            
                # Ensure columns match the exact order the model expects
                expected_cols = model.feature_names_in_
                input_df = input_df.reindex(columns=expected_cols, fill_value=0.0)
            
                # 3. Make Prediction using baseline 0.05 threshold
                probs = model.predict_proba(input_df)[0][1]
                is_fraud = probs >= 0.05
            
                # 4. Display Results
                if is_fraud:
                    st.error(f"🚨 **FRAUD ALERT!!** This transaction has been flagged as highly suspicious. (Risk Score: {probs:.1%})", icon="🚨")
                else:
                    st.success(f"✅ **CLEAR.** This transaction appears to be legitimate. (Risk Score: {probs:.1%})", icon="✅")
            else:
                st.warning("⚠️ Model not found! Please run main.py first.")
else:
    st.info("👈 Please enter transaction details in the sidebar and click 'Scan Transaction'.")