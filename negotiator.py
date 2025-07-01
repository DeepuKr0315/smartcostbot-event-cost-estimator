# negotiator.py - AI-powered negotiation helper
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# Use the correct model ID from HuggingFace
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"


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
    prompt = f"""
You are {data['vendor_name']}, an experienced event vendor negotiating with {data['planner_name']} from {data['company_name']} about organizing a {data['event_type']} on {data['event_date']}.

Here's what they've shared:
- Total estimated cost: ₹{data['total_cost']}
- Event type: {data['event_type']}
- Season: {"Peak" if data['season'] == "yes" else "Off-Peak"}
- Services requested: {', '.join(data['services'])}

Please offer a fair discount and final price in a friendly way.
Make sure to keep the tone professional but open to negotiation.
Include a realistic closing with your name/company.

Vendor Response:
"""
    return prompt


def fetch_vendor_offer(prompt, tokenizer, model):
    """
    Gets a response from the AI vendor using the local LLM.
    """
    print("💬 Talking to vendor AI...")
    inputs = tokenizer(prompt, return_tensors="pt")

    # Move input tensors to the same device as the model
    device = model.device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Clean up output
    if "Vendor Response:" in response:
        response = response.split("Vendor Response:")[-1].strip()

    return response