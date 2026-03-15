"""
Evaluation test script for Challenge 3: Vehicle Pricing >3.5t

Runs all training samples against the /solve endpoint and scores results
using the same rubric as the challenge:
  - Price accuracy: 50% (±10% = full, degrades to ±50%)
  - Sources:        25% (3+ = full, fewer = proportional)
  - Price range:    15% (expected value within min-max)
  - Currency:       10% (must be "CZK")

Usage:
    python test_eval.py                     # test all 60 samples
    python test_eval.py --limit 5           # test first 5 only
    python test_eval.py --index 0           # test a single sample
    python test_eval.py --parallel 5        # run 5 tests concurrently
    python test_eval.py --url http://...    # custom endpoint URL
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

DEFAULT_URL = "http://localhost:8080/solve"
TEST_DATA_FILE = "test_data.json"


def score_price_accuracy(actual, expected):
    """50% weight: ±10% = 1.0, linear decay to 0.0 at ±50%."""
    if actual is None or expected is None or expected == 0:
        return 0.0
    pct_error = abs(actual - expected) / expected
    if pct_error <= 0.10:
        return 1.0
    elif pct_error >= 0.50:
        return 0.0
    else:
        # Linear decay from 1.0 at 10% to 0.0 at 50%
        return 1.0 - (pct_error - 0.10) / 0.40


def score_sources(sources):
    """25% weight: 3+ sources = 1.0, fewer = proportional."""
    if not sources:
        return 0.0
    count = len(sources)
    if count >= 3:
        return 1.0
    return count / 3.0


def score_price_range(result, expected_value):
    """15% weight: expected value falls within returned min-max range."""
    if expected_value is None:
        return 0.0
    price_range = result.get("price_range_czk", {})
    rmin = price_range.get("min")
    rmax = price_range.get("max")
    if rmin is None or rmax is None:
        return 0.0
    if rmin <= expected_value <= rmax:
        return 1.0
    return 0.0


def score_currency(result):
    """10% weight: must be 'CZK'."""
    return 1.0 if result.get("currency") == "CZK" else 0.0


def evaluate_one(result, expected_output):
    """Score a single result against expected output. Returns dict of scores."""
    expected_value = expected_output.get("estimated_value_czk")
    actual_value = result.get("estimated_value_czk")

    s_accuracy = score_price_accuracy(actual_value, expected_value)
    s_sources = score_sources(result.get("sources", []))
    s_range = score_price_range(result, expected_value)
    s_currency = score_currency(result)

    weighted = (
        s_accuracy * 0.50
        + s_sources * 0.25
        + s_range * 0.15
        + s_currency * 0.10
    )

    return {
        "price_accuracy": s_accuracy,
        "sources": s_sources,
        "price_range": s_range,
        "currency": s_currency,
        "weighted_total": weighted,
        "actual_value": actual_value,
        "expected_value": expected_value,
    }


def run_test(url, test_case, index):
    """Send a request and evaluate the response."""
    inp = test_case["input"]
    expected = test_case["expected_output"]

    label = f"{inp.get('make', '?')} {inp.get('model', '?')} ({inp.get('year', '?')}) - {inp.get('body_type', '?')}"
    print(f"\n[{index}] {label}")
    print(f"    Expected: {expected.get('estimated_value_czk'):,} CZK")

    try:
        start = time.time()
        resp = httpx.post(url, json=inp, timeout=120)
        elapsed = time.time() - start
        result = resp.json()
    except Exception as e:
        print(f"    ERROR: {e}")
        return None, 0

    scores = evaluate_one(result, expected)

    actual = scores["actual_value"]
    actual_str = f"{actual:,}" if actual else "None"
    pct_err = ""
    if actual and expected.get("estimated_value_czk"):
        err = (actual - expected["estimated_value_czk"]) / expected["estimated_value_czk"] * 100
        pct_err = f" ({err:+.1f}%)"

    print(f"    Got:      {actual_str} CZK{pct_err}")
    print(f"    Sources:  {len(result.get('sources', []))}")
    print(f"    Range:    {result.get('price_range_czk', {}).get('min', '?'):,} - {result.get('price_range_czk', {}).get('max', '?'):,}")
    print(f"    Scores:   accuracy={scores['price_accuracy']:.2f}  sources={scores['sources']:.2f}  range={scores['price_range']:.2f}  currency={scores['currency']:.2f}")
    print(f"    TOTAL:    {scores['weighted_total']:.2f} / 1.00  ({elapsed:.1f}s)")

    return scores, elapsed


def main():
    parser = argparse.ArgumentParser(description="Evaluate vehicle pricing endpoint")
    parser.add_argument("--url", default=DEFAULT_URL, help="Endpoint URL")
    parser.add_argument("--limit", type=int, default=None, help="Max test cases to run")
    parser.add_argument("--index", type=int, default=None, help="Run a single test case by index")
    parser.add_argument("--data", default=TEST_DATA_FILE, help="Test data JSON file")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay in seconds between requests (default: 1.0)")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel workers (default: 1)")
    args = parser.parse_args()

    with open(args.data) as f:
        test_cases = json.load(f)

    if args.index is not None:
        test_cases = [test_cases[args.index]]
        indices = [args.index]
    elif args.limit:
        test_cases = test_cases[: args.limit]
        indices = list(range(len(test_cases)))
    else:
        indices = list(range(len(test_cases)))

    print(f"Running {len(test_cases)} test cases against {args.url} (workers={args.parallel})")
    print("=" * 70)

    all_scores = []
    total_time = 0

    if args.parallel > 1:
        # Parallel execution
        results_map = {}
        start_total = time.time()
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {}
            for i, tc in zip(indices, test_cases):
                f = executor.submit(run_test, args.url, tc, i)
                futures[f] = i
            for f in as_completed(futures):
                idx = futures[f]
                scores, elapsed = f.result()
                results_map[idx] = (scores, elapsed)

        # Print results in order
        for i in indices:
            scores, elapsed = results_map[i]
            if scores:
                all_scores.append(scores)
            total_time += elapsed
        total_time = time.time() - start_total
    else:
        # Sequential execution
        for i, tc in zip(indices, test_cases):
            scores, elapsed = run_test(args.url, tc, i)
            if scores:
                all_scores.append(scores)
            total_time += elapsed
            if i != indices[-1]:
                time.sleep(args.delay)

    # Summary
    print("\n" + "=" * 70)
    print(f"SUMMARY: {len(all_scores)}/{len(test_cases)} successful")
    print(f"Total time: {total_time:.1f}s (avg {total_time/max(len(test_cases),1):.1f}s per request)")

    if all_scores:
        avg_accuracy = sum(s["price_accuracy"] for s in all_scores) / len(all_scores)
        avg_sources = sum(s["sources"] for s in all_scores) / len(all_scores)
        avg_range = sum(s["price_range"] for s in all_scores) / len(all_scores)
        avg_currency = sum(s["currency"] for s in all_scores) / len(all_scores)
        avg_total = sum(s["weighted_total"] for s in all_scores) / len(all_scores)

        perfect = sum(1 for s in all_scores if s["price_accuracy"] >= 1.0)
        good = sum(1 for s in all_scores if s["price_accuracy"] >= 0.75)
        bad = sum(1 for s in all_scores if s["price_accuracy"] < 0.25)

        print(f"\nAverage scores:")
        print(f"  Price accuracy (50%): {avg_accuracy:.3f}")
        print(f"  Sources (25%):        {avg_sources:.3f}")
        print(f"  Price range (15%):    {avg_range:.3f}")
        print(f"  Currency (10%):       {avg_currency:.3f}")
        print(f"  ─────────────────────────────")
        print(f"  WEIGHTED TOTAL:       {avg_total:.3f} / 1.000")
        print(f"\nAccuracy breakdown:")
        print(f"  Perfect (±10%): {perfect}/{len(all_scores)}")
        print(f"  Good (±20%):    {good}/{len(all_scores)}")
        print(f"  Bad (>±40%):    {bad}/{len(all_scores)}")

        # Worst performers
        worst = sorted(all_scores, key=lambda s: s["price_accuracy"])[:5]
        print(f"\nWorst 5 by accuracy:")
        for s in worst:
            exp = s["expected_value"]
            act = s["actual_value"]
            if exp and act:
                err = (act - exp) / exp * 100
                print(f"  Expected {exp:>10,}  Got {act:>10,}  ({err:+.1f}%)  score={s['price_accuracy']:.2f}")
            else:
                print(f"  Expected {exp}  Got {act}  score={s['price_accuracy']:.2f}")

    # Check metrics
    try:
        metrics_url = args.url.replace("/solve", "/metrics")
        r = httpx.get(metrics_url, timeout=5)
        m = r.json()
        print(f"\nGemini token usage:")
        print(f"  Requests: {m.get('gemini_request_count', '?')}")
        print(f"  Total tokens: {m.get('total_tokens', '?')}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
