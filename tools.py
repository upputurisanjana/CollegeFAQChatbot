"""
tools.py — BVRIT Chatbot Function Calling (Exercises 1-5)
"""
import json, os, re
from datetime import date, datetime
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 1 — Tool JSON Schemas
# ─────────────────────────────────────────────────────────────────────────────

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
                    "annual_fee":     {"type": "number", "description": "Annual tuition fee in INR from BVRIT documents."},
                    "years":          {"type": "number", "description": "Number of years (1-6). B.Tech=4, M.Tech=2."},
                    "scholarship_pct":{"type": "number", "description": "Scholarship % to deduct (0-100). Pass 0 if none."},
                    "hostel_annual":  {"type": "number", "description": "Optional annual hostel fee in INR."},
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


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 2 — fee_calculator implementation
# EXERCISE 3 — Edge case validation (E1-E5)
# ─────────────────────────────────────────────────────────────────────────────

def fee_calculator(annual_fee: float, years: float,
                   scholarship_pct: float, hostel_annual: float = 0.0) -> dict:
    """
    Compute total BVRIT fees with optional scholarship and hostel.
    Includes full E1-E5 edge case validation (Exercise 3).
    """
    errors = []

    # E1: Zero or negative years
    if years <= 0:
        errors.append(f"'years' must be at least 1, got {years}. "
                      "Please specify a valid programme duration (e.g. 4 for B.Tech).")

    # E1 continued: unreasonably large years
    if years > 6:
        errors.append(f"'years' value {years} is unusually high. "
                      "BVRIT programmes are 2 (M.Tech) or 4 (B.Tech) years.")

    # E3: Impossible scholarship percentage
    if scholarship_pct < 0 or scholarship_pct > 100:
        errors.append(f"'scholarship_pct' must be between 0 and 100, got {scholarship_pct}. "
                      "A scholarship cannot exceed 100% or be negative.")

    # E2: Negative fee (contradictory / nonsense input)
    if annual_fee <= 0:
        errors.append(f"'annual_fee' must be positive, got {annual_fee}.")

    # E4: Prompt injection guard — fees shouldn't be astronomically large
    if annual_fee > 10_000_000:
        errors.append(f"'annual_fee' value {annual_fee} looks incorrect. "
                      "BVRIT fees are typically under ₹2,00,000/year.")

    # E5: hostel_annual sanity check
    if hostel_annual < 0:
        errors.append(f"'hostel_annual' cannot be negative, got {hostel_annual}.")

    if errors:
        return {"error": " | ".join(errors)}

    # Clamp years to integer for cleaner output
    years = int(years)

    discount        = annual_fee * (scholarship_pct / 100)
    net_annual      = annual_fee - discount
    tuition_total   = net_annual * years
    hostel_total    = hostel_annual * years
    grand_total     = tuition_total + hostel_total

    result = {
        "annual_fee_before_scholarship": annual_fee,
        "scholarship_pct": scholarship_pct,
        "scholarship_amount_per_year": round(discount, 2),
        "net_annual_tuition": round(net_annual, 2),
        "years": years,
        "total_tuition": round(tuition_total, 2),
    }
    if hostel_annual > 0:
        result["annual_hostel_fee"]  = hostel_annual
        result["total_hostel"]       = round(hostel_total, 2)
        result["grand_total"]        = round(grand_total, 2)
    else:
        result["total_cost"] = round(tuition_total, 2)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 4 — date_checker implementation
# ─────────────────────────────────────────────────────────────────────────────

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
        "event_name":     event_name,
        "target_date":    target_date,
        "today":          str(today),
        "status":         status,
        "days_remaining": delta,
        "message":        message,
    }


# ─────────────────────────────────────────────────────────────────────────────
# percentage_calculator implementation
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Tool dispatcher
# ─────────────────────────────────────────────────────────────────────────────

TOOL_FUNCTIONS = {
    "fee_calculator":       fee_calculator,
    "date_checker":         date_checker,
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


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 2 + 4 — Chatbot with tool routing
# ─────────────────────────────────────────────────────────────────────────────

def ask_bvrit(question: str, tools_enabled: list[dict] = None) -> dict:
    """
    Single-loop BVRIT chatbot with function calling.
    Returns: {path, tool_name, arguments, tool_result, answer}

    Routing paths:
      1. tool_call  → model returned a tool call → execute → re-submit → get answer
      2. rag        → model returned text (no tool) → return as-is
      3. none       → conversational, no retrieval needed
    """
    import openai

    api_key  = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE") or "https://openrouter.ai/api/v1"
    client   = openai.OpenAI(api_key=api_key, base_url=api_base)
    model_id = "deepseek/deepseek-r1"

    if tools_enabled is None:
        tools_enabled = TOOLS

    # Pull RAG context for grounding
    try:
        from rag import retrieve, _expand_query, _format_context, MIN_RELEVANCE_SCORE
        chunks  = retrieve(_expand_query(question), None, top_k=5)
        relevant = [c for c in chunks if c.score >= MIN_RELEVANCE_SCORE] or chunks[:3]
        context = _format_context(relevant)
    except Exception:
        context = "(Knowledge base unavailable — answer from tool only if applicable.)"

    system_msg = (
        "You are the BVRIT Hyderabad FAQ assistant. "
        "Answer ONLY using the CONTEXT below or by calling the provided tools. "
        "If the question requires fee arithmetic, call fee_calculator. "
        "If it asks about dates/deadlines, call date_checker. "
        "If it is purely conversational, reply directly without tools.\n\n"
        f"CONTEXT:\n{context}"
    )

    messages = [
        {"role": "system",  "content": system_msg},
        {"role": "user",    "content": question},
    ]

    # Step 1 — first LLM call (may return tool_call or text)
    response = client.chat.completions.create(
        model       = model_id,
        messages    = messages,
        tools       = tools_enabled,
        tool_choice = "auto",
        temperature = 0.1,
        max_tokens  = 600,
    )

    choice      = response.choices[0]
    finish      = choice.finish_reason
    tool_name   = None
    tool_args   = {}
    tool_result = None

    # Path 1: tool call
    if finish == "tool_calls" or (choice.message.tool_calls):
        tc        = choice.message.tool_calls[0]
        tool_name = tc.function.name
        tool_args = json.loads(tc.function.arguments)
        tool_result = dispatch_tool(tool_name, tool_args)

        # Step 2 — feed result back to model for final answer
        messages.append(choice.message)           # assistant turn with tool_call
        messages.append({
            "role":         "tool",
            "tool_call_id": tc.id,
            "content":      tool_result,
        })
        response2 = client.chat.completions.create(
            model       = model_id,
            messages    = messages,
            temperature = 0.1,
            max_tokens  = 600,
        )
        answer = response2.choices[0].message.content or ""
        path   = "tool_call"

    # Path 2/3: plain text
    else:
        answer = choice.message.content or ""
        # Heuristic: if answer references document facts it used RAG
        path = "rag" if context not in ("(Knowledge base unavailable — answer from tool only if applicable.)",) \
                        and len(answer) > 40 else "none"

    return {
        "question":    question,
        "path":        path,
        "tool_name":   tool_name,
        "arguments":   tool_args,
        "tool_result": json.loads(tool_result) if tool_result else None,
        "answer":      answer,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 5 — 10-query test suite
# ─────────────────────────────────────────────────────────────────────────────

TEST_QUERIES = [
    {"id": 1,  "question": "What B.Tech branches does BVRIT offer?",                               "expected_path": "rag"},
    {"id": 2,  "question": "What is the annual tuition for ECE?",                                  "expected_path": "rag"},
    {"id": 3,  "question": "What's the total 4-year tuition for ECE?",                             "expected_path": "tool_call"},
    {"id": 4,  "question": "If I get a 25% scholarship on CSE tuition, what's my annual fee?",    "expected_path": "tool_call"},
    {"id": 5,  "question": "Is the admission deadline past?",                                      "expected_path": "tool_call"},
    {"id": 6,  "question": "How many days until the semester exam?",                               "expected_path": "tool_call"},
    {"id": 7,  "question": "What's the total cost for 4 years: tuition + hostel?",                "expected_path": "tool_call"},
    {"id": 8,  "question": "Tell me about the campus facilities.",                                 "expected_path": "rag"},
    {"id": 9,  "question": "Thanks, that's helpful!",                                              "expected_path": "none"},
    {"id": 10, "question": "Calculate my total 4-year cost with 20% scholarship on tuition.",     "expected_path": "tool_call"},
]

AGENT_REFLECTION = """
Queries 7 and 10 expose the core limit of single-loop function calling:
both require TWO retrieval steps (get tuition fee → get hostel fee → compute)
but the single-loop pattern only allows one tool call per turn.
In query 7 the model must guess or hallucinate the hostel fee if it wasn't
in the RAG context alongside tuition. In query 10 it must retrieve the fee,
compute the scholarship deduction, then recompute the total — steps that
must happen sequentially, each depending on the previous result.

A Day 6 agent solves this with a planning loop: it can issue 'retrieve tuition fee',
observe the result, issue 'retrieve hostel fee', observe again, then call
fee_calculator with both values. The agent maintains state across steps and
decides when it has gathered enough information to produce the final answer —
something a single-turn function call cannot do.
"""


def run_test_suite(verbose: bool = True) -> list[dict]:
    """Run all 10 test queries and return results table."""
    results = []
    for t in TEST_QUERIES:
        if verbose:
            print(f"\n[{t['id']:02d}] {t['question']}")
        try:
            r = ask_bvrit(t["question"])
        except Exception as e:
            r = {"path": "error", "tool_name": None, "arguments": {}, "answer": str(e)}

        passed = r["path"] == t["expected_path"]
        row = {
            "id":            t["id"],
            "question":      t["question"],
            "expected_path": t["expected_path"],
            "actual_path":   r["path"],
            "tool_called":   r.get("tool_name") or "—",
            "arguments":     r.get("arguments") or {},
            "answer":        (r.get("answer") or "")[:120],
            "pass":          "PASS" if passed else "FAIL",
        }
        results.append(row)
        if verbose:
            status = "✅" if passed else "❌"
            print(f"  Path: {r['path']} (expected {t['expected_path']}) {status}")
            if r.get("tool_name"):
                print(f"  Tool: {r['tool_name']}  Args: {r.get('arguments')}")
            print(f"  Answer: {row['answer'][:80]}...")
    return results


def print_results_table(results: list[dict]) -> None:
    print("\n" + "="*100)
    print(f"{'ID':>3} | {'Expected':12} | {'Actual':12} | {'Tool':22} | {'Pass':5} | Question")
    print("-"*100)
    for r in results:
        print(f"{r['id']:>3} | {r['expected_path']:12} | {r['actual_path']:12} | "
              f"{r['tool_called']:22} | {r['pass']:5} | {r['question'][:45]}")
    passed = sum(1 for r in results if r["pass"] == "PASS")
    print("="*100)
    print(f"Result: {passed}/{len(results)} PASS")
    print("\nAGENT REFLECTION:")
    print(AGENT_REFLECTION)


# ─────────────────────────────────────────────────────────────────────────────
# CLI — run all exercises
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode in ("1", "schemas", "all"):
        print("\n=== EXERCISE 1: Tool Schemas ===")
        for t in TOOLS:
            print(f"  Tool: {t['function']['name']}")
            print(f"  Desc: {t['function']['description'][:80]}...")

    if mode in ("2", "fee", "all"):
        print("\n=== EXERCISE 2: fee_calculator test ===")
        ex2_queries = [
            ("What departments does BVRIT have?",                       "RAG only"),
            ("What's the total tuition for 4 years of B.Tech CSE?",    "RAG + fee_calculator"),
            ("Hello, how are you?",                                     "none"),
            ("If I get a 15% scholarship, what's my annual CSE fee?",  "RAG + fee_calculator"),
        ]
        for q, expected in ex2_queries:
            r = ask_bvrit(q, tools_enabled=[TOOLS[0]])  # fee_calculator only
            print(f"  Q: {q[:55]}")
            print(f"     Path={r['path']} | Tool={r['tool_name']} | Expected≈{expected}")

    if mode in ("3", "edge", "all"):
        print("\n=== EXERCISE 3: Edge Cases ===")
        edge_cases = [
            ("E1", {"annual_fee": 120000, "years": 0,   "scholarship_pct": 0}),
            ("E2", {"annual_fee": 120000, "years": 4,   "scholarship_pct": 0}),   # contradictory handled by RAG
            ("E3", {"annual_fee": 120000, "years": 4,   "scholarship_pct": 150}),
            ("E4", {"annual_fee": 999999999, "years": 4,"scholarship_pct": 0}),
            ("E5", {"annual_fee": 120000, "years": 4,   "scholarship_pct": 0, "hostel_annual": -5000}),
        ]
        for label, args in edge_cases:
            result = fee_calculator(**args)
            print(f"  {label}: {result}")

    if mode in ("4", "date", "all"):
        print("\n=== EXERCISE 4: date_checker test ===")
        date_queries = [
            "When is the last date for EAMCET counselling?",
            "Is the EAMCET counselling deadline already past?",
            "How many days until the semester exam?",
            "What's the total 4-year hostel cost?",
            "What departments does BVRIT have?",
            "Hi there",
        ]
        for q in date_queries:
            r = ask_bvrit(q, tools_enabled=TOOLS[:2])  # fee + date
            print(f"  Q: {q[:55]}")
            print(f"     Path={r['path']} | Tool={r['tool_name']}")

    if mode in ("5", "suite", "all"):
        print("\n=== EXERCISE 5: Full 10-Query Test Suite ===")
        results = run_test_suite(verbose=True)
        print_results_table(results)
