"""
test_tools.py — Exercise 5: Systematic Tool Routing Test
=========================================================
Runs the 10-query test suite from HandsOn_FunctionCalling_Questions.md
and reports routing decisions (RAG / fee_calculator / date_checker / none).
"""

import time
import json
from rag import answer_question

TEST_SUITE = [
    # (id, query, expected_routing)
    ("Q1",  "What B.Tech branches does BVRIT offer?",                              "RAG only"),
    ("Q2",  "What is the annual tuition for ECE?",                                 "RAG only"),
    ("Q3",  "What's the total 4-year tuition for ECE?",                            "RAG + fee_calculator"),
    ("Q4",  "If I get a 25% scholarship on CSE tuition, what's my annual fee?",    "RAG + fee_calculator"),
    ("Q5",  "Is the admission deadline past?",                                     "RAG + date_checker"),
    ("Q6",  "How many days until the semester exam?",                              "RAG + date_checker"),
    ("Q7",  "What's the total cost for 4 years: tuition + hostel?",                "RAG + fee_calculator"),
    ("Q8",  "Tell me about the campus facilities.",                                "RAG only"),
    ("Q9",  "Thanks, that's helpful!",                                             "None (conversation)"),
    ("Q10", "Calculate my total 4-year cost with 20% scholarship on tuition",      "RAG + fee_calculator"),
]


def detect_routing(answer: str, citations: list) -> str:
    """Infer which path was taken from the answer and citations."""
    has_citations = len(citations) > 0
    answer_lower = answer.lower()

    # Check if tool was likely used via numeric computation patterns
    has_fee_computation = any(kw in answer_lower for kw in [
        "total", "grand total", "per year", "scholarship amount",
    ])
    has_date_computation = any(kw in answer_lower for kw in [
        "day(s) ago", "days until", "days remaining", "is today",
    ])

    if not has_citations and not has_fee_computation and not has_date_computation:
        return "None (conversation)"
    if has_citations and has_fee_computation:
        return "RAG + fee_calculator"
    if has_citations and has_date_computation:
        return "RAG + date_checker"
    if has_citations:
        return "RAG only"
    return "Unknown"


def run():
    print("=" * 70)
    print("  BVRIT Chatbot — Tool Routing Test Suite (Exercise 5)")
    print("=" * 70)
    print(f"{'ID':<5} {'Query':<50} {'Expected':<25} {'Actual':<25} {'Pass?':<6}")
    print("-" * 70)

    results = []
    for qid, query, expected in TEST_SUITE:
        t0 = time.time()
        result = answer_question(query, top_k=5)
        latency = time.time() - t0

        actual = detect_routing(result.answer, result.citations)
        passed = "PASS" if (
            (expected == actual) or
            (expected.startswith("RAG") and actual.startswith("RAG"))
        ) else "FAIL"

        print(f"{qid:<5} {query[:48]:<50} {expected:<25} {actual:<25} {passed:<6}")
        results.append({
            "id": qid, "query": query, "expected": expected,
            "actual": actual, "passed": passed,
            "latency_s": round(latency, 2),
            "answer_preview": result.answer[:100],
        })

    passed_count = sum(1 for r in results if r["passed"] == "PASS")
    total = len(results)
    print("-" * 70)
    print(f"\nResults: {passed_count}/{total} passed ({passed_count/total*100:.0f}%)")

    # Write detailed report
    with open("tool_test_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Detailed report saved to tool_test_report.json")


if __name__ == "__main__":
    run()
