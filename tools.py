"""
tools.py - BVRIT Chatbot Function Calling Tools
================================================
Three tools exposed to the LLM via OpenAI function-calling:
  - fee_calculator       : multi-year tuition/hostel cost with scholarship
  - date_checker         : compare an academic date against today
  - percentage_calculator: percentage arithmetic for BVRIT figures

Public interface:
    from tools import TOOLS, dispatch_tool
"""
import json
from datetime import date, datetime

# ---------------------------------------------------------------------------
# Tool JSON Schemas
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fee_calculator",
            "description": (
                "Calculate total BVRIT Hyderabad tuition or hostel fees across multiple "
                "years, or apply a scholarship discount to a BVRIT annual fee. Use ONLY "
                "for BVRIT fee arithmetic: total cost over N years, annual fee after "
                "scholarship deduction, or tuition+hostel combined totals. "
                "Do NOT use for date comparisons, placement percentages, or general math."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "annual_fee":      {"type": "number", "description": "Annual tuition fee in INR from BVRIT documents."},
                    "years":           {"type": "number", "description": "Number of years (1-6). B.Tech=4, M.Tech=2."},
                    "scholarship_pct": {"type": "number", "description": "Scholarship % to deduct (0-100). Pass 0 if none."},
                    "hostel_annual":   {"type": "number", "description": "Optional annual hostel fee in INR."},
                },
                "required": ["annual_fee", "years", "scholarship_pct"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "date_checker",
            "description": (
                "Compare a BVRIT academic date (admission deadline, exam date, counselling "
                "cutoff) against today and return whether it is past, today, or upcoming "
                "with days remaining. Use ONLY when the user asks if a BVRIT deadline has "
                "passed or how many days until a BVRIT event. "
                "Do NOT use for fee calculations or scholarship math."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_date": {"type": "string", "description": "Date in YYYY-MM-DD format extracted from BVRIT documents."},
                    "event_name":  {"type": "string", "description": "Human-readable event name, e.g. 'EAMCET counselling deadline'."},
                },
                "required": ["target_date", "event_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "percentage_calculator",
            "description": (
                "Calculate a percentage in BVRIT academic contexts: scholarship amount "
                "as a percentage of fee, placement rate from student counts, or cutoff "
                "conversions. Use when user asks 'what is X% of Y' for BVRIT figures. "
                "Do NOT use for multi-year fee totals (use fee_calculator) or dates."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "value":      {"type": "number", "description": "Base value (e.g. annual fee, students placed)."},
                    "percentage": {"type": "number", "description": "Percentage to compute (0-100)."},
                    "operation":  {"type": "string", "enum": ["of", "what_pct"],
                                   "description": "'of'=compute pct% of value. 'what_pct'=what % is value of total."},
                    "total":      {"type": "number", "description": "Denominator for 'what_pct' operation."},
                },
                "required": ["value", "percentage", "operation"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# fee_calculator
# ---------------------------------------------------------------------------

def fee_calculator(annual_fee: float, years: float,
                   scholarship_pct: float, hostel_annual: float = 0.0) -> dict:
    """
    Compute total BVRIT fees with optional scholarship and hostel.
    Edge-case validation:
      E1: years <= 0 or > 6
      E2: annual_fee <= 0
      E3: scholarship_pct outside 0-100
      E4: annual_fee unreasonably large (>10M)
      E5: hostel_annual < 0
    """
    errors = []

    if years <= 0:
        errors.append(f"'years' must be at least 1, got {years}. "
                      "Please specify a valid programme duration (e.g. 4 for B.Tech).")
    if years > 6:
        errors.append(f"'years' value {years} is unusually high. "
                      "BVRIT programmes are 2 (M.Tech) or 4 (B.Tech) years.")
    if scholarship_pct < 0 or scholarship_pct > 100:
        errors.append(f"'scholarship_pct' must be between 0 and 100, got {scholarship_pct}.")
    if annual_fee <= 0:
        errors.append(f"'annual_fee' must be positive, got {annual_fee}.")
    if annual_fee > 10_000_000:
        errors.append(f"'annual_fee' value {annual_fee} looks incorrect. "
                      "BVRIT fees are typically under Rs 2,00,000/year.")
    if hostel_annual < 0:
        errors.append(f"'hostel_annual' cannot be negative, got {hostel_annual}.")

    if errors:
        return {"error": " | ".join(errors)}

    years = int(years)
    discount      = annual_fee * (scholarship_pct / 100)
    net_annual    = annual_fee - discount
    tuition_total = net_annual * years
    hostel_total  = hostel_annual * years
    grand_total   = tuition_total + hostel_total

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
        result["total_hostel"]      = round(hostel_total, 2)
        result["grand_total"]       = round(grand_total, 2)
    else:
        result["total_cost"] = round(tuition_total, 2)

    return result

# ---------------------------------------------------------------------------
# date_checker
# ---------------------------------------------------------------------------

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
        status  = "past"
        message = f"The {event_name} was {abs(delta)} day(s) ago ({target_date})."
    elif delta == 0:
        status  = "today"
        message = f"The {event_name} is TODAY ({target_date})."
    else:
        status  = "upcoming"
        message = f"The {event_name} is in {delta} day(s), on {target_date}."

    return {
        "event_name":     event_name,
        "target_date":    target_date,
        "today":          str(today),
        "status":         status,
        "days_remaining": delta,
        "message":        message,
    }

# ---------------------------------------------------------------------------
# percentage_calculator
# ---------------------------------------------------------------------------

def percentage_calculator(value: float, percentage: float,
                           operation: str, total: float = 0.0) -> dict:
    if percentage < 0 or percentage > 100:
        return {"error": f"percentage must be 0-100, got {percentage}."}
    if operation == "of":
        result = value * (percentage / 100)
        return {
            "value": value, "percentage": percentage,
            "result": round(result, 2),
            "message": f"{percentage}% of {value} = {round(result, 2)}",
        }
    elif operation == "what_pct":
        if total <= 0:
            return {"error": "total must be > 0 for 'what_pct' operation."}
        result = (value / total) * 100
        return {
            "value": value, "total": total,
            "result": round(result, 2),
            "message": f"{value} is {round(result, 2)}% of {total}",
        }
    return {"error": f"Unknown operation '{operation}'."}

# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

TOOL_FUNCTIONS = {
    "fee_calculator":        fee_calculator,
    "date_checker":          date_checker,
    "percentage_calculator": percentage_calculator,
}


def dispatch_tool(tool_name: str, arguments: dict) -> str:
    """Execute a tool call and return JSON string result."""
    fn = TOOL_FUNCTIONS.get(tool_name)
    if not fn:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    try:
        result = fn(**arguments)
        return json.dumps(result, ensure_ascii=False)
    except TypeError as e:
        return json.dumps({"error": f"Bad arguments for {tool_name}: {e}"})


# ---------------------------------------------------------------------------
# Quick self-test (python tools.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== fee_calculator ===")
    print(fee_calculator(120000, 4, 25, 60000))
    print(fee_calculator(120000, 0, 25))       # E1: years=0
    print(fee_calculator(120000, 4, 150))      # E3: bad scholarship

    print("\n=== date_checker ===")
    print(date_checker("2025-01-01", "Admission deadline"))
    print(date_checker("2027-01-01", "Semester exam"))
    print(date_checker("bad-date", "Test"))

    print("\n=== percentage_calculator ===")
    print(percentage_calculator(120000, 25, "of"))
    print(percentage_calculator(480, 600, "what_pct", total=600))
