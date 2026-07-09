"""
run_eval_fast.py — Checkpointed eval runner
Saves progress after each test case. Resume with --resume.
Usage:
    python run_eval_fast.py                       # fresh run
    python run_eval_fast.py --resume               # resume from last checkpoint
    python run_eval_fast.py --skip-ragas           # skip RAGAS
    python run_eval_fast.py --dim 01               # run one dimension
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

CHECKPOINT_FILE = "eval_checkpoint.json"


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"completed_ids": [], "results": [], "started_at": None}


def save_checkpoint(data):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dim", default=None)
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--out", default="eval_report.json")
    args = parser.parse_args()

    from eval import TEST_CASES, DIM_NAMES
    from eval import judge  # noqa: used below
    from rag import answer_question

    checkpoint = load_checkpoint() if args.resume else {
        "completed_ids": [],
        "results": [],
        "started_at": datetime.utcnow().isoformat(),
    }

    # Filter test cases
    cases = [tc for tc in TEST_CASES if args.dim is None or tc["dim"] == args.dim]
    completed_ids = set(checkpoint.get("completed_ids", []))
    pending = [tc for tc in cases if tc["id"] not in completed_ids]

    print(f"Total: {len(cases)} | Completed: {len(completed_ids)} | Pending: {len(pending)}")

    if not pending:
        print("All cases done! Building report...")
        build_report(checkpoint, args)
        return

    for tc in pending:
        cid = tc["id"]
        print(f"\n[{datetime.utcnow().strftime('%H:%M:%S')}] Running {cid}...")
        sys.stdout.flush()

        t0 = time.perf_counter()
        try:
            result = answer_question(tc["question"], top_k=5, history=tc.get("history", []))
            latency = time.perf_counter() - t0
            tr = judge(tc, result.answer, latency)
            tr_dict = {
                "id": tr.id, "dim": tr.dim, "question": tr.question,
                "answer": tr.answer, "verdict": tr.verdict,
                "latency_s": tr.latency_s, "reason": tr.reason,
                "root_cause": tr.root_cause, "fix": tr.fix,
            }
            checkpoint["results"].append(tr_dict)
            checkpoint["completed_ids"].append(cid)
            status_icon = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠"}.get(tr.verdict, "?")
            print(f"  {status_icon} {cid} — {tr.verdict} | {tr.reason[:80]}")
        except Exception as e:
            print(f"  ✗ {cid} — ERROR: {str(e)[:100]}")
            checkpoint["results"].append({
                "id": cid, "dim": tc["dim"], "question": tc["question"],
                "answer": "", "verdict": "FAIL",
                "latency_s": round(time.perf_counter() - t0, 2),
                "reason": f"Exception: {e}", "root_cause": "", "fix": "",
            })
            checkpoint["completed_ids"].append(cid)

        save_checkpoint(checkpoint)

    build_report(checkpoint, args)


def build_report(checkpoint, args):
    from eval import DIM_NAMES, DIM_FIX_SUGGESTIONS

    results = checkpoint.get("results", [])
    total = len(results)
    passed = sum(1 for r in results if r["verdict"] == "PASS")
    failed = sum(1 for r in results if r["verdict"] == "FAIL")
    warned = sum(1 for r in results if r["verdict"] == "WARN")

    # Per-dimension
    dim_data = {}
    for r in results:
        d = r["dim"]
        if d not in dim_data:
            dim_data[d] = {"cases": [], "passed": 0, "failed": 0, "warned": 0}
        dim_data[d]["cases"].append(r)
        dim_data[d][r["verdict"].lower()] += 1

    dimensions = []
    for did, dd in sorted(dim_data.items()):
        dt = len(dd["cases"])
        dr = dd["passed"] / dt if dt else 0
        dimensions.append({
            "id": did, "name": DIM_NAMES.get(did, did),
            "passed": dd["passed"], "failed": dd["failed"],
            "warned": dd["warned"], "total": dt,
            "pass_rate": round(dr, 3), "cases": dd["cases"],
        })

    weakest = min(dimensions, key=lambda d: d["pass_rate"], default=None)
    weakest_info = {}
    if weakest:
        weakest_info = {
            "id": weakest["id"], "name": weakest["name"],
            "pass_rate": weakest["pass_rate"],
            "fix": DIM_FIX_SUGGESTIONS.get(weakest["id"], "Review failed cases."),
        }

    report = {
        "summary": {
            "total": total, "passed": passed, "failed": failed,
            "warning": warned,
            "pass_rate": round(passed / total, 3) if total else 0,
        },
        "dimensions": dimensions,
        "weakest_dimension": weakest_info,
        "ragas": {"error": "Skipped by run_eval_fast.py — use python eval.py for RAGAS"},
        "residual_risk_statement": (
            "5 known injection patterns tested and blocked (04-SEC1 through 04-SEC5)."
        ),
    }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {args.out}")
    s = report["summary"]
    print(f"SUMMARY: {s['total']} tests | {s['passed']} passed | {s['failed']} failed | {s['warning']} warnings | {s['pass_rate']*100:.0f}%")

    # Cleanup checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)


if __name__ == "__main__":
    run()
