# app.py - Streamlit Web Interface for SmartCostBot (with Chat)
import streamlit as st
from estimator import calculate_total_cost
from negotiator import load_negotiation_model, generate_negotiation_prompt, fetch_vendor_offer
import pandas as pd

# Page config
st.set_page_config(page_title="SmartCostBot – Event Cost Estimator", layout="centered")

st.title("💼 SmartCostBot")
st.subheader("Event Cost Estimator & AI Negotiator")
st.markdown("Built with ❤️ for UtsavAi Internship Round 1")

# Sidebar inputs
df = pd.read_csv("mock_services.csv")
service_list = list(df['Service'])

with st.sidebar:
    st.header("📝 Enter Event Details")
    services = st.multiselect("Select Services", service_list)
    guests = st.number_input("Number of Guests", min_value=1, value=100)
    event_type = st.text_input("Event Type (e.g., Wedding, Corporate)")
    season = st.radio("Is this during peak season?", ["yes", "no"])
    vendor_name = st.text_input("Vendor Name / Company", placeholder="Enter your vendor name here")
    planner_name = st.text_input("Your Name (Event Planner)", placeholder="Enter your name here")
    company_name = st.text_input("Client / Company Name (optional)", placeholder="Enter client name here")
    event_date = st.date_input("Event Date", value=pd.to_datetime("today"))

if st.button("💰 Estimate Cost & Get Initial Offer"):
    if not services:
        st.warning("⚠️ Please select at least one service.")
    else:
        with st.spinner("🧠 Calculating cost and negotiating offer..."):
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
            initial_response = fetch_vendor_offer(prompt, tokenizer, model)

            # Save context and response in session state
            st.session_state.breakdown = breakdown
            st.session_state.total = total
            st.session_state.initial_response = initial_response
            st.session_state.data = data
            st.session_state.tokenizer = tokenizer
            st.session_state.model = model
            st.session_state.response_context = f"""
Initial Estimate: ₹{total}
Services Selected: {', '.join(services)}
Event Type: {event_type}
Event Date: {event_date}
Season: {"Peak" if season == 'yes' else "Off-Peak"}
Vendor Name: {vendor_name}
Client Name: {company_name}

Vendor Offer:
{initial_response}
"""

            st.success("✅ Done! Scroll down to view results.")

# Display cost breakdown
if "breakdown" in st.session_state:
    st.markdown("### 📊 Estimated Cost Breakdown")
    for item, price in st.session_state.breakdown.items():
        st.markdown(f"- **{item}**: ₹{price}")
    st.markdown(f"### 🏷️ **Total Estimated Cost**: ₹{st.session_state.total}")

    st.markdown("### 💬 Initial Vendor Response")
    st.write(st.session_state.initial_response)

# --- Chat-based Negotiation Section ---
st.markdown("## 💬 Live Vendor Chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask the vendor about pricing, discounts, or services..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("💬 Talking to vendor AI..."):
            # Build full prompt with context
            context = st.session_state.get("response_context", "")
            full_prompt = f"""
You are {st.session_state.data['vendor_name']}, an experienced event vendor.
The planner just asked: "{prompt}"

Use the following context to understand the event:
{context}

Respond naturally and professionally. Keep it concise and helpful.
"""

            # Fetch response
            tokenizer = st.session_state.tokenizer
            model = st.session_state.model
            inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7, do_sample=True)
            ai_response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            st.markdown(ai_response)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": ai_response})

# Footer
st.markdown("---")
st.markdown("Made with 💡 using [Streamlit](https://streamlit.io ) | Powered by TinyLlama")