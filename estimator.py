# estimator.py - Cost calculator for event services
# Written by [Your Name] for UtsavAi Internship Round 1

import pandas as pd


def calculate_total_cost(services, guests):
    """
    Calculates total cost based on selected services and number of guests.
    Prices are adjusted slightly per guest (₹10 per head).
    """
    df = pd.read_csv("mock_services.csv")
    total = 0
    breakdown = {}

    for service in services:
        row = df[df['Service'] == service]
        if not row.empty:
            base_price = int(row['Base_Price'].values[0])
            min_price = int(row['Min_Price'].values[0])
            max_price = int(row['Max_Price'].values[0])

            # Adjust price based on number of guests
            adjusted_price = base_price + (guests * 10)
            adjusted_price = max(min_price, min(adjusted_price, max_price))

            total += adjusted_price
            breakdown[service] = adjusted_price

    return total, breakdown