# cli.py - My Event Cost Tool
# Written by [Your Name] for UtsavAi Internship Round 1

from estimator import calculate_total_cost
from negotiator import load_negotiation_model, generate_negotiation_prompt, fetch_vendor_offer
import pandas as pd


def ask_for_event_details():
    print("Welcome to SmartCostBot – Your DIY Event Budget Tool\n")

    # Load available services from CSV
    df = pd.read_csv("mock_services.csv")
    service_list = list(df['Service'])

    print("Available Services:")
    for idx, service in enumerate(service_list):
        print(f"{idx + 1}. {service}")

    while True:
        try:
            choice_input = input("\nEnter service numbers separated by commas (e.g., 1,3,5): ")
            selected_indices = [int(num.strip()) - 1 for num in choice_input.split(",")]
            break
        except ValueError:
            print("⚠️ Invalid input. Please enter numbers separated by commas.")

    chosen_services = [service_list[i] for i in selected_indices]
    guest_count = int(input("How many guests are expected? "))
    event_type = input("What kind of event is this? (e.g., Wedding, Birthday, Corporate): ")
    season_status = input("Is this during peak season? (yes/no): ").lower()

    # NEW FIELDS
    vendor_name = input("Enter vendor name/company: ")
    planner_name = input("Enter your name (event planner): ")
    company_name = input("Enter client/company name (optional): ")
    event_date = input("Enter event date (e.g., 2025-12-25): ")

    return {
        'services': chosen_services,
        'guests': guest_count,
        'event_type': event_type,
        'season': season_status,
        'vendor_name': vendor_name,
        'planner_name': planner_name,
        'company_name': company_name,
        'event_date': event_date
    }


def run_cost_analysis():
    user_data = ask_for_event_details()
    services = user_data['services']
    guests = user_data['guests']

    total, breakdown = calculate_total_cost(services, guests)

    print("\n--- Estimated Cost Breakdown ---")
    for item, price in breakdown.items():
        print(f"- {item}: ₹{price}")
    print(f"\nTotal Estimated Cost: ₹{total}")

    print("\nTrying to negotiate a better deal...\n")

    # Add calculated total to user_data for prompt generation
    user_data['total_cost'] = total

    # Load model once
    tokenizer, model = load_negotiation_model()

    prompt = generate_negotiation_prompt(user_data)
    response = fetch_vendor_offer(prompt, tokenizer, model)
    print(response)


if __name__ == "__main__":
    run_cost_analysis()