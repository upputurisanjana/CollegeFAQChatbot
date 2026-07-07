"""
calculators.py — Fee, date, and percentage utilities for BVRIT chatbot
======================================================================
Extracted from the disconnected tools.py function-calling chatbot.
"""

import json
from datetime import date, datetime


def fee_calculator(annual_fee: float, years: float,
                   scholarship_pct: float, hostel_annual: float = 0.0) -> dict:
    """
    Compute total BVRIT fees with optional scholarship and hostel.
    """
    errors = []

    if years <= 0:
        errors.append(f"'years' must be at least 1, got {years}. "
                      "Please specify a valid programme duration (e.g. 4 for B.Tech).")
    if years > 6:
        errors.append(f"'years' value {years} is unusually high. "
                      "BVRIT programmes are 2 (M.Tech) or 4 (B.Tech) years.")
    if scholarship_pct < 0 or scholarship_pct > 100:
        errors.append(f"'scholarship_pct' must be between 0 and 100, got {scholarship_pct}")
    if annual_fee <= 0:
        errors.append(f"'annual_fee' must be positive, got {annual_fee}.")
    if annual_fee > 10_000_000:
        errors.append(f"'annual_fee' value {annual_fee} looks incorrect. "
                      "BVRIT fees are typically under ₹2,00,000/year.")
    if hostel_annual < 0:
        errors.append(f"'hostel_annual' cannot be negative, got {hostel_annual}.")

    if errors:
        return {"error": " | ".join(errors)}

    years = int(years)
    discount = annual_fee * (scholarship_pct / 100)
    net_annual = annual_fee - discount
    tuition_total = net_annual * years
    hostel_total = hostel_annual * years
    grand_total = tuition_total + hostel_total

    result = {
        "annual_fee_before_scholarship": annual_fee,
        "scholarship_pct": scholarship_pct,
        "scholarship_amount_per_year": round(discount, 2),
        "net_annual_tuition": round(net_annual, 2),
        "years": years,
        "total_tuition": round(tuition_total, 2),
    }
    if hostel_annual > 0:
        result["annual_hostel_fee"] = hostel_annual
        result["total_hostel"] = round(hostel_total, 2)
        result["grand_total"] = round(grand_total, 2)
    else:
        result["total_cost"] = round(tuition_total, 2)

    return result


def date_checker(target_date: str, event_name: str) -> dict:
    """
    Compare target_date (YYYY-MM-DD) against today.
    Returns status: 'past' | 'today' | 'upcoming', plus days_remaining.
    """
    try:
        target = datetime.strptime(target_date, "%Y-%m-%d").date()
    except ValueError:
        return {"error": f"Invalid date format '{target_date}'. Use YYYY-MM-DD."}

    today = date.today()
    delta = (target - today).days

    if delta < 0:
        status = "past"
        message = f"The {event_name} was {abs(delta)} day(s) ago ({target_date})."
    elif delta == 0:
        status = "today"
        message = f"The {event_name} is TODAY ({target_date})."
    else:
        status = "upcoming"
        message = f"The {event_name} is in {delta} day(s), on {target_date}."

    return {
        "event_name": event_name,
        "target_date": target_date,
        "today": str(today),
        "status": status,
        "days_remaining": delta,
        "message": message,
    }


def percentage_calculator(value: float, percentage: float,
                          operation: str, total: float = 0.0) -> dict:
    if percentage < 0 or percentage > 100:
        return {"error": f"percentage must be 0-100, got {percentage}."}
    if operation == "of":
        result = value * (percentage / 100)
        return {"value": value, "percentage": percentage,
                "result": round(result, 2),
                "message": f"{percentage}% of {value} = {round(result, 2)}"}
    elif operation == "what_pct":
        if total <= 0:
            return {"error": "total must be > 0 for 'what_pct' operation."}
        result = (value / total) * 100
        return {"value": value, "total": total,
                "result": round(result, 2),
                "message": f"{value} is {round(result, 2)}% of {total}"}
    return {"error": f"Unknown operation '{operation}'."}
