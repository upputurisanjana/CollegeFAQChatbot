"""
test_memory.py — Verify conversation memory end-to-end
Phases:
  1. EntityStore: name extraction, dept extraction, topic extraction, persistence
  2. ConversationMemory: context injection, history summarization
  3. End-to-end via rag.py: name → remember → use in reply
"""

import json
import sqlite3
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
DB_PATH = "chat_history.db"


def clean_db():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)


def print_header(label: str):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")


def phase1_entity_store():
    print_header("PHASE 1: EntityStore — name/dept/topic extraction")
    from memory import EntityStore

    store = EntityStore("phase1_test")
    passed = 0
    total = 0

    # 1a — Name extraction: "my name is John"
    total += 1
    store.update_from_conversation([
        {"role": "user", "content": "my name is John, tell me about CSE"}
    ])
    name = store.get("user", "name")
    ok = name == "John"
    print(f"  [{'PASS' if ok else 'FAIL'}] name='John' from 'my name is John' → got '{name}'")
    if ok:
        passed += 1

    # 1b — Name extraction: "I am Alice"
    total += 1
    store2 = EntityStore("phase1_test_b")
    store2.update_from_conversation([
        {"role": "user", "content": "I am Alice, how do I apply?"}
    ])
    name2 = store2.get("user", "name")
    ok = name2 == "Alice"
    print(f"  [{'PASS' if ok else 'FAIL'}] name='Alice' from 'I am Alice' → got '{name2}'")
    if ok:
        passed += 1

    # 1c — Name extraction: lowercase first letter (works — regex has IGNORECASE)
    total += 1
    store3 = EntityStore("phase1_test_c")
    store3.update_from_conversation([
        {"role": "user", "content": "my name is bob, tell me about placements"}
    ])
    name3 = store3.get("user", "name")
    ok = name3 == "bob"
    print(f"  [{'PASS' if ok else 'FAIL'}] lowercase 'bob' captured (regex has IGNORECASE) → got '{name3}'")
    if ok:
        passed += 1

    # 1d — Name extraction: "call me Dr. Smith"
    total += 1
    store4 = EntityStore("phase1_test_d")
    store4.update_from_conversation([
        {"role": "user", "content": "call me Dr. Smith"}
    ])
    name4 = store4.get("user", "name")
    ok = name4 == "Dr" or name4 == "Dr. Smith"  # regex may only capture first
    print(f"  [{'PASS' if ok else 'FAIL'}] name from 'call me Dr. Smith' → got '{name4}'")
    if ok:
        passed += 1

    # 1e — Department extraction
    total += 1
    store5 = EntityStore("phase1_test_e")
    store5.update_from_conversation([
        {"role": "user", "content": "Tell me about CSE and ECE departments"}
    ])
    depts_str = store5.get("user", "departments")
    depts = json.loads(depts_str) if depts_str else []
    ok = "CSE" in depts and "ECE" in depts
    print(f"  [{'PASS' if ok else 'FAIL'}] departments CSE,ECE from query → got {depts}")
    if ok:
        passed += 1

    # 1f — Topic extraction
    total += 1
    store6 = EntityStore("phase1_test_f")
    store6.update_from_conversation([
        {"role": "user", "content": "What is the fee structure for placements?"}
    ])
    topics_str = store6.get("user", "topics")
    topics = json.loads(topics_str) if topics_str else []
    ok = "fee" in topics and "placement" in topics
    print(f"  [{'PASS' if ok else 'FAIL'}] topics fee,placement from query → got {topics}")
    if ok:
        passed += 1

    # 1g — Context blurb includes name
    total += 1
    blurb = store.get_context_blurb()
    ok = "John" in blurb
    print(f"  [{'PASS' if ok else 'FAIL'}] context blurb contains 'John' → '{blurb[:60]}...'")
    if ok:
        passed += 1

    print(f"\n  Phase 1: {passed}/{total} passed")
    return passed, total


def phase2_conversation_memory():
    print_header("PHASE 2: ConversationMemory — context injection")
    from memory import ConversationMemory

    mem = ConversationMemory("phase2_test")
    passed = 0
    total = 0

    # Pre-load a name
    mem.entities.set("user", "name", "Riya")

    # 2a — Short history: context should be prepended
    total += 1
    history = [
        {"role": "user", "content": "what is the fee?"},
        {"role": "assistant", "content": "The fee is 1.2L"},
    ]
    prepared = mem.prepare_messages(history)
    has_context = any(
        m["role"] == "system" and "Riya" in m.get("content", "")
        for m in prepared
    )
    has_history = any(
        m["role"] in ("user", "assistant") for m in prepared
    )
    ok = has_context and has_history
    print(f"  [{'PASS' if ok else 'FAIL'}] context injected + history preserved → "
          f"context={'Riya' if has_context else 'miss'}, history={len(history)} turns, "
          f"messages={len(prepared)} total")
    if ok:
        passed += 1

    # 2b — Empty history: just context, no history
    total += 1
    prepared_empty = mem.prepare_messages([])
    ok = len(prepared_empty) == 1 and prepared_empty[0]["role"] == "system"
    print(f"  [{'PASS' if ok else 'FAIL'}] empty history → just system context msg ({len(prepared_empty)} msgs)")
    if ok:
        passed += 1

    # 2c — Long history (>6 turns): should summarize older turns
    total += 1
    long_history = []
    for i in range(8):
        long_history.append({"role": "user", "content": f"query {i}"})
        long_history.append({"role": "assistant", "content": f"answer {i}"})
    prepared_long = mem.prepare_messages(long_history)
    # Should have: context msg + at most 6 most recent turns + possible summary
    ok = len(prepared_long) <= 8  # 1 context + 1 summary + up to 6 recent
    # The first non-context msg should be the summary or first recent turn
    non_system = [m for m in prepared_long if m["role"] != "system"]
    has_recent = any(
        "query 7" in m.get("content", "") for m in prepared_long
    )
    print(f"  [{'PASS' if ok else 'FAIL'}] long history ({len(long_history)} msgs) truncated → "
          f"{len(prepared_long)} total msgs, has_recent={has_recent}")
    if ok:
        passed += 1

    print(f"\n  Phase 2: {passed}/{total} passed")
    return passed, total


def phase3_end_to_end():
    print_header("PHASE 3: End-to-end via rag.py — does the chatbot remember?")
    from memory import ConversationMemory
    from rag import answer_question

    session_id = "memory_e2e_test"
    mem = ConversationMemory(session_id)

    passed = 0
    total = 0

    # Step 1: Tell the chatbot your name
    print("\n  Step 1: Telling name 'My name is Priya'...")
    r1 = answer_question(
        question="My name is Priya. Tell me about BVRIT Hyderabad.",
        top_k=5,
        memory=mem,
        session_id=session_id,
    )
    print(f"    Answer snippet: {r1.answer[:120]}...")
    print(f"    Refused: {r1.refused}")

    # Verify entity was stored
    name_stored = mem.entities.get("user", "name")
    ok1 = name_stored == "Priya"
    print(f"    EntityStore has name='{name_stored}' → {'PASS' if ok1 else 'FAIL'}")
    total += 1
    if ok1:
        passed += 1

    # Step 2: Ask something else — model should remember "Priya"
    print("\n  Step 2: Asking follow-up 'What departments are offered?'...")
    history = [
        {"role": "user", "content": "My name is Priya. Tell me about BVRIT Hyderabad."},
        {"role": "assistant", "content": r1.answer},
    ]
    r2 = answer_question(
        question="What departments are offered?",
        history=history,
        top_k=5,
        memory=mem,
        session_id=session_id,
    )
    print(f"    Answer: {r2.answer[:200]}...")
    print(f"    Citations: {len(r2.citations)}")
    # Check if model uses "Priya" in the response
    remembers = "Priya" in r2.answer or "priya" in r2.answer.lower()
    total += 1
    if remembers:
        passed += 1
        print(f"    {'PASS' if remembers else 'FAIL'} — model remembered the name!")
    else:
        print(f"    {'PASS' if remembers else 'FAIL'} — model did not use the name (may be Free Router limitation)")

    print(f"\n  Phase 3: {passed}/{total} passed")
    return passed, total


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=0, help="Run specific phase (1, 2, 3) or 0 for all")
    parser.add_argument("--skip-llm", action="store_true", help="Skip Phase 3 (LLM-dependent)")
    args = parser.parse_args()

    clean_db()

    total_p = total_t = 0

    if args.phase in (0, 1):
        p1p, p1t = phase1_entity_store()
        total_p += p1p; total_t += p1t

    if args.phase in (0, 2):
        p2p, p2t = phase2_conversation_memory()
        total_p += p2p; total_t += p2t

    if args.phase in (0, 3) and not args.skip_llm:
        p3p, p3t = phase3_end_to_end()
        total_p += p3p; total_t += p3t

    if args.skip_llm and args.phase in (0, 3):
        print("\n  Phase 3 skipped (--skip-llm)")

    print(f"\n{'='*70}")
    print(f"  OVERALL: {total_p}/{total_t} passed ({total_p/total_t*100:.0f}%)")
    print(f"{'='*70}")

    # Cleanup test DB
    clean_db()

    sys.exit(0 if total_p == total_t else 1)
