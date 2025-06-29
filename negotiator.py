# negotiator.py - AI-powered negotiation helper
# Written by [Your Name] for UtsavAi Internship Round 1

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# Use the correct model ID from HuggingFace
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def load_negotiation_model():
    """Load tokenizer and model with device mapping"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto"
    )
    return tokenizer, model


def generate_negotiation_prompt(data):
    """
    Creates a prompt for the AI vendor to respond to.
    Simulates a realistic vendor negotiation scenario with personalization.
    """
    total_cost = data['total_cost']
    event_type = data['event_type']
    season = "Peak" if data['season'] == "yes" else "Off-Peak"
    vendor_name = data['vendor_name']
    planner_name = data['planner_name']
    company_name = data['company_name'] or "Valued Client"
    event_date = data['event_date']

    prompt = f"""
You are {vendor_name}, an experienced event vendor.  
{planner_name}, an event planner from {company_name}, has reached out about organizing a {event_type} on {event_date}.  

Here are the details:
- Estimated cost: ₹{total_cost}
- Event type: {event_type}
- Season: {season}

Respond as the vendor offering a fair and realistic discount based on the season and event size.  
Include the final amount after discount.  
Use a friendly but professional tone.  
Do not use placeholders like [Your Name].  
Make sure to keep the message concise and relevant.

Vendor Response:
"""
    return prompt


def fetch_vendor_offer(prompt, tokenizer, model):
    print("💬 Talking to vendor AI...")
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    output = model.generate(**inputs, max_new_tokens=350, do_sample=True, temperature=0.9, top_k=50)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract only what the model generated after the prompt
    if prompt in decoded:
        response = decoded.split(prompt)[-1].strip()
    else:
        response = decoded.strip()

    return response