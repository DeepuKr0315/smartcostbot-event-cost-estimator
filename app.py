# app.py - Streamlit Web Interface for SmartCostBot
import streamlit as st
from estimator import calculate_total_cost
from negotiator import load_negotiation_model, generate_negotiation_prompt, fetch_vendor_offer
import pandas as pd

# Page config
st.set_page_config(page_title="SmartCostBot – Event Cost Estimator", layout="centered")

st.title("💼 SmartCostBot")
st.subheader("Event Cost Estimator & AI Negotiator")
st.markdown("Built with ❤️ for UtsavAi Internship Round 1")

# Load services
df = pd.read_csv("mock_services.csv")
service_list = list(df['Service'])

# Sidebar inputs
with st.sidebar:
    st.header("📝 Enter Event Details")
    services = st.multiselect("Select Services", service_list)
    guests = st.number_input("Number of Guests", min_value=1, value=100)
    event_type = st.text_input("Event Type (e.g., Wedding, Corporate)", placeholder="Corporate")
    season = st.radio("Is this during peak season?", ["yes", "no"])
    vendor_name = st.text_input("Vendor Name / Company", placeholder="Your Vendor Name")
    planner_name = st.text_input("Your Name (Event Planner)", placeholder="Planner Name")
    company_name = st.text_input("Client / Company Name (optional)", placeholder="Valued Client")
    event_date = st.date_input("Event Date", value=pd.to_datetime("2025-07-01"))

# Main area
if st.button("💰 Estimate Cost & Negotiate"):
    if not services:
        st.warning("⚠️ Please select at least one service.")
    else:
        with st.spinner("💬 Talking to vendor AI... Please wait a moment."):
            # Step 1: Calculate cost
            total, breakdown = calculate_total_cost(services, guests)

            # Step 2: Prepare data for negotiation
            data = {
                'services': services,
                'guests': guests,
                'event_type': event_type,
                'season': season,
                'vendor_name': vendor_name,
                'planner_name': planner_name,
                'company_name': company_name,
                'event_date': str(event_date),
                'total_cost': total
            }

            # Step 3: Load model and get response
            tokenizer, model = load_negotiation_model()
            prompt = generate_negotiation_prompt(data)
            response = fetch_vendor_offer(prompt, tokenizer, model)

        # Show result after spinner ends
        st.markdown("### 📊 Estimated Cost Breakdown")
        for item, price in breakdown.items():
            st.markdown(f"- **{item}**: ₹{price}")
        st.markdown(f"### 🏷️ **Total Estimated Cost**: ₹{total}")

        st.markdown("### 💬 Vendor Negotiation Response")
        st.write(response)
else:
    st.info("👉 Select services and click **Estimate Cost & Negotiate**.")

# Footer
st.markdown("---")
st.markdown("Made with 💡 using [Streamlit](https://streamlit.io ) | Powered by TinyLlama")