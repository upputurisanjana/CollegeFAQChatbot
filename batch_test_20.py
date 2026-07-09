"""
batch_test_20.py — 20-question end-to-end pipeline test
Tests: retrieval quality, answer format, correctness, refusal
Usage:
    python batch_test_20.py                    # retrieval + generation
    python batch_test_20.py --retrieval-only   # fast: no LLM calls
    python batch_test_20.py --out results.json
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

QUESTIONS = [
    # Departments (2)
    "What departments does BVRIT Hyderabad offer?",
    "Tell me about the CSE department",
    # Faculty (2)
    "List CSE department faculty",
    "Who is the principal of BVRIT?",
    # Admissions (2)
    "How do I apply for admission to BVRIT?",
    "What is the B-Category admission process?",
    # Fees (3)
    "What is the tuition fee for CSE?",
    "What is the hostel fee?",
    "Tell me about the PM Vidyalaxmi scheme",
    # Placements (2)
    "What is the placement record at BVRIT?",
    "Which companies recruit from BVRIT?",
    # Facilities (2)
    "Tell me about the library",
    "Is there a hostel on campus?",
    # Student Activities (2)
    "What student clubs are available?",
    "Tell me about the sports club",
    # Research (2)
    "What research is done at BVRIT?",
    "How many patents has BVRIT filed?",
    # Contact (2)
    "How can I contact BVRIT?",
    "What is the address of BVRIT Hyderabad?",
    # Out-of-scope (1)
    "What is the canteen menu today?",
]


def test_retrieval():
    """Fast: test retrieval only (no LLM calls)"""
    from rag import retrieve, MIN_RELEVANCE_SCORE

    print(f"{'#':>3} {'Question':58s} {'Chunks':6s} {'TopScore':8s} {'Pass':5s}")
    print("-" * 85)

    passed = 0
    results = []
    for i, q in enumerate(QUESTIONS, 1):
        t0 = time.perf_counter()
        chunks = retrieve(q, None, top_k=5)
        elapsed = time.perf_counter() - t0
        scores = [c.score for c in chunks]
        top = max(scores) if scores else 0
        n_above = sum(1 for s in scores if s >= MIN_RELEVANCE_SCORE)
        ok = n_above >= 1
        if ok: passed += 1
        print(f"{i:>3} {q[:56]:58s} {len(chunks):3d}  {top:.4f}  {'PASS' if ok else 'FAIL'}")
        results.append({
            "id": i, "question": q, "chunks": len(chunks),
            "top_score": round(top, 4), "n_above_threshold": n_above,
            "passed": ok, "latency_s": round(elapsed, 3),
        })

    print("-" * 85)
    print(f"Retrieval: {passed}/{len(QUESTIONS)} passed ({passed/len(QUESTIONS)*100:.0f}%)")
    return results


def test_generation():
    """Full end-to-end: retrieval + LLM generation"""
    from rag import answer_question

    out_of_scope_idx = len(QUESTIONS)  # last question (i is 1-indexed)

    print(f"\n{'#':>3} {'Question':55s} {'AnsLen':7s} {'Cites':6s} {'Ref':4s} {'Lat':5s} {'Pass':5s}")
    print("-" * 95)

    passed = 0
    results = []
    for i, q in enumerate(QUESTIONS, 1):
        t0 = time.perf_counter()
        try:
            r = answer_question(q, top_k=5)
            elapsed = time.perf_counter() - t0
            ans_len = len(r.answer)
            cites = len(r.citations)
            refused = r.refused

            if i == out_of_scope_idx:
                # Out-of-scope must be refused
                ok = refused
            else:
                # In-scope: must have substantive answer AND citations
                ok = ans_len > 20 and cites > 0 and not refused

            if ok: passed += 1
            print(f"{i:>3} {q[:53]:55s} {ans_len:4d}c  {cites:3d}   {str(refused):4s} {elapsed:4.1f}s {'PASS' if ok else 'FAIL'}")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"{i:>3} {q[:53]:55s} {'ERROR':7s} {'':6s} {'':4s} {elapsed:4.1f}s {'FAIL':5s}")
            results.append({"id": i, "question": q, "error": str(e), "passed": False})
            continue

        results.append({
            "id": i, "question": q, "answer_len": ans_len,
            "citations": cites, "refused": refused,
            "latency_s": round(elapsed, 2), "passed": ok,
            "answer_preview": r.answer[:120],
        })

    print("-" * 95)
    print(f"Generation: {passed}/{len(QUESTIONS)} passed ({passed/len(QUESTIONS)*100:.0f}%)")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--out", default="batch_test_20_results.json")
    args = parser.parse_args()

    print(f"Running {'retrieval-only' if args.retrieval_only else 'full end-to-end'} test on {len(QUESTIONS)} questions\n")

    if args.retrieval_only:
        results = test_retrieval()
    else:
        results_r = test_retrieval()
        results_g = test_generation()
        results = {"retrieval": results_r, "generation": results_g}

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.out}")
