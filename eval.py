#!/usr/bin/env python3
"""
QuoteParser Automated Eval
--------------------------
Runs all 5 vendor quote PDFs through the extraction prompt,
scores outputs against ground truth, and prints a report.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python eval.py
    python eval.py --prompt prompts/v2.txt   # test a different prompt
    python eval.py --model claude-haiku-4-5-20251001  # test faster model
"""

import anthropic
import base64
import json
import time
import argparse
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────

PDF_DIR = Path("/mnt/user-data/uploads")
GROUND_TRUTH_FILE = Path("/home/claude/ground_truth.json")

DEFAULT_MODEL = "claude-sonnet-4-20250514"
COST_PER_1M_INPUT  = 3.00   # USD — Sonnet
COST_PER_1M_OUTPUT = 15.00  # USD — Sonnet
LATENCY_BUDGET_S   = 7.0

DEFAULT_PROMPT = """You are a vendor quote extraction agent for a B2B fintech platform. Extract all information from this vendor quote PDF and return ONLY a valid JSON object. No preamble, no markdown fences, just raw JSON.

Schema:
{
  "quote_metadata": { "quote_id": string|null, "quote_date": string|null, "valid_until": string|null, "payment_terms": string|null },
  "vendor": { "name": string|null, "address": string|null, "phone": string|null, "email": string|null },
  "customer": { "name": string|null, "contact_name": string|null, "address": string|null, "phone": string|null, "email": string|null },
  "deal_structure": { "is_subscription": boolean, "subscription_term_months": number|null, "billing_frequency": "annual"|"monthly"|"one-time"|null, "auto_renew": boolean|null },
  "line_items": [{ "line_number": number, "item_name": string, "product_id": string|null, "sku": string|null, "quantity": number, "unit_price": number, "extended_price": number, "classification": "hardware"|"software"|"service"|"labor"|"discount"|"tax"|"shipping", "is_financeable": boolean, "notes": string|null }],
  "financials": { "subtotal": number|null, "discount_total": number|null, "shipping": number|null, "tax": number|null, "grand_total": number|null, "financeable_total": number|null, "currency": "USD" },
  "confidence": { "financeable_total": number (0-1), "classification": number (0-1), "deal_term": number (0-1), "reasoning": string },
  "flags": [{ "type": "warning"|"info", "field": string, "message": string }]
}

Classification rules: hardware=physical devices/cables/accessories; software=licenses/subscriptions/SaaS; service=support contracts/maintenance; labor=professional services/installation; discount=credits/negative lines; tax=tax charges; shipping=freight.

is_financeable: hardware=true; software/licenses=true; multi-year service contracts=true; monthly service=false; labor=false; discount/tax/shipping=false.

financeable_total = sum of extended_price where is_financeable=true.

Flag any: ambiguous pricing (e.g. two price columns), items that may not be financeable, missing critical fields, calculation discrepancies, subscription terms needing human confirmation."""

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class DocResult:
    file: str
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    extracted: dict = field(default_factory=dict)
    error: Optional[str] = None

    # Scored fields (filled after scoring)
    grand_total_match: Optional[bool] = None
    grand_total_extracted: Optional[float] = None
    grand_total_actual: Optional[float] = None
    financeable_total_match: Optional[bool] = None
    financeable_total_extracted: Optional[float] = None
    financeable_total_actual: Optional[float] = None
    subscription_match: Optional[bool] = None
    term_match: Optional[bool] = None
    vendor_name_match: Optional[bool] = None
    customer_name_match: Optional[bool] = None
    quote_id_match: Optional[bool] = None
    classification_score: Optional[float] = None
    flags_raised: int = 0
    avg_confidence: Optional[float] = None
    within_budget: bool = False
    edge_case_flags: list = field(default_factory=list)

# ── Extraction ────────────────────────────────────────────────────────────────

def load_pdf_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def extract(client: anthropic.Anthropic, pdf_b64: str, prompt: str, model: str) -> tuple[dict, float, int, int]:
    start = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=4000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_b64}},
                {"type": "text", "text": prompt}
            ]
        }]
    )
    latency = time.time() - start
    text = response.content[0].text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    data = json.loads(text)
    return data, latency, response.usage.input_tokens, response.usage.output_tokens

# ── Scoring ───────────────────────────────────────────────────────────────────

def fuzzy_match_name(extracted: Optional[str], actual: str) -> bool:
    if not extracted:
        return False
    e = extracted.lower().strip()
    a = actual.lower().strip()
    # Check if key words from actual appear in extracted
    key_words = [w for w in a.split() if len(w) > 3]
    return any(w in e for w in key_words)

def total_match(extracted: Optional[float], actual: Optional[float], tolerance: float = 0.10) -> bool:
    if extracted is None or actual is None:
        return False
    if actual == 0:
        return extracted == 0
    return abs(extracted - actual) / actual <= tolerance

def score_classifications(extracted_items: list, ground_truth_classifications: dict) -> float:
    if not extracted_items or not ground_truth_classifications:
        return None
    correct = 0
    total = 0
    for item in extracted_items:
        name = item.get("item_name", "").lower()
        for gt_key, gt_class in ground_truth_classifications.items():
            if any(word in name for word in gt_key.lower().split()[:3] if len(word) > 3):
                total += 1
                if item.get("classification") == gt_class:
                    correct += 1
                break
    return round(correct / total, 2) if total > 0 else None

def check_edge_case_flags(result: DocResult, gt: dict) -> list:
    flags = []
    doc_flags = [f.get("message", "").lower() for f in result.extracted.get("flags", [])]
    notes = gt.get("notes", "").lower()

    if "two price columns" in notes or "msrp" in notes:
        flagged = any("msrp" in f or "price column" in f or "discounted" in f for f in doc_flags)
        flags.append(("Dual price column detected", flagged))

    if "annualised" in notes or "annual" in notes and "5 yr" in notes:
        flagged = any("annual" in f or "5 year" in f or "total contract" in f or "annuali" in f for f in doc_flags)
        flags.append(("Annualised EA pricing flagged", flagged))

    if "labor" in notes and "not financeable" in notes:
        labor_items = [i for i in result.extracted.get("line_items", []) if i.get("classification") == "labor"]
        labor_not_financeable = all(not i.get("is_financeable", True) for i in labor_items) if labor_items else False
        flags.append(("Labor correctly excluded from financeable", labor_not_financeable))

    if "subscription" in notes and "nrc" in notes:
        flagged = any("nrc" in f or "non-recurring" in f or "implementation" in f for f in doc_flags)
        flags.append(("NRC vs recurring charges distinguished", flagged))

    return flags

def score_result(result: DocResult, gt: dict) -> DocResult:
    if result.error:
        return result

    ex = result.extracted
    fin_gt = gt.get("financials", {})
    meta_gt = gt.get("quote_metadata", {})
    ds_gt = gt.get("deal_structure", {})

    # Totals
    result.grand_total_actual = fin_gt.get("grand_total")
    result.grand_total_extracted = ex.get("financials", {}).get("grand_total")
    result.grand_total_match = total_match(result.grand_total_extracted, result.grand_total_actual)

    result.financeable_total_actual = fin_gt.get("financeable_total")
    result.financeable_total_extracted = ex.get("financials", {}).get("financeable_total")
    result.financeable_total_match = total_match(result.financeable_total_extracted, result.financeable_total_actual)

    # Deal structure
    result.subscription_match = ex.get("deal_structure", {}).get("is_subscription") == ds_gt.get("is_subscription")
    term_ex = ex.get("deal_structure", {}).get("subscription_term_months")
    term_gt = ds_gt.get("subscription_term_months")
    result.term_match = (term_ex == term_gt) or (term_gt is None and term_ex is None)

    # Names
    result.vendor_name_match = fuzzy_match_name(ex.get("vendor", {}).get("name"), gt.get("vendor", {}).get("name", ""))
    result.customer_name_match = fuzzy_match_name(ex.get("customer", {}).get("name"), gt.get("customer", {}).get("name", ""))

    # Quote ID
    qid_ex = ex.get("quote_metadata", {}).get("quote_id")
    qid_gt = meta_gt.get("quote_id")
    result.quote_id_match = (qid_ex == qid_gt) or (qid_gt is None)

    # Classifications
    gt_classes = gt.get("classifications", {})
    if isinstance(gt_classes, dict) and gt_classes:
        result.classification_score = score_classifications(ex.get("line_items", []), gt_classes)

    # Flags & confidence
    result.flags_raised = len(ex.get("flags", []))
    conf = ex.get("confidence", {})
    scores = [v for k, v in conf.items() if k != "reasoning" and isinstance(v, (int, float))]
    result.avg_confidence = round(sum(scores) / len(scores), 2) if scores else None

    # Latency
    result.within_budget = result.latency_s <= LATENCY_BUDGET_S

    # Edge cases
    result.edge_case_flags = check_edge_case_flags(result, gt)

    return result

# ── Reporting ──────────────────────────────────────────────────────────────────

def tick(val: Optional[bool]) -> str:
    if val is True:  return "✓"
    if val is False: return "✗"
    return "—"

def print_report(results: list[DocResult], model: str, prompt_version: str):
    print("\n")
    print("=" * 72)
    print(f"  QUOTEPARSER EVAL REPORT")
    print(f"  Model: {model}   Prompt: {prompt_version}")
    print("=" * 72)

    total_cost = sum(r.cost_usd for r in results if not r.error)
    avg_latency = sum(r.latency_s for r in results if not r.error) / max(len([r for r in results if not r.error]), 1)
    within_budget = sum(1 for r in results if r.within_budget and not r.error)

    print(f"\n  {'Document':<38} {'Latency':>8} {'Budget':>7} {'Tokens':>8} {'Cost':>8} {'G.Total':>8} {'Fin.Total':>10} {'Sub':>5} {'Term':>5} {'Vendor':>7} {'Cust':>6}")
    print(f"  {'-'*38} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*5} {'-'*5} {'-'*7} {'-'*6}")

    for r in results:
        name = r.file.replace("_", " ").replace(".pdf", "")[:37]
        if r.error:
            print(f"  {name:<38} {'ERROR: ' + r.error[:40]}")
            continue
        tokens = r.input_tokens + r.output_tokens
        print(
            f"  {name:<38}"
            f" {r.latency_s:>7.1f}s"
            f" {'✓' if r.within_budget else '✗':>7}"
            f" {tokens:>8,}"
            f" ${r.cost_usd:>6.4f}"
            f"  {tick(r.grand_total_match):>7}"
            f"  {tick(r.financeable_total_match):>9}"
            f"  {tick(r.subscription_match):>4}"
            f"  {tick(r.term_match):>4}"
            f"  {tick(r.vendor_name_match):>6}"
            f"  {tick(r.customer_name_match):>5}"
        )

    print(f"\n  {'SUMMARY':<38} {avg_latency:>7.1f}s {within_budget}/5 {'':>8} ${total_cost:>6.4f}")

    # Accuracy breakdown
    print(f"\n  ACCURACY BREAKDOWN")
    print(f"  {'─'*50}")

    checks = [
        ("Grand total extracted correctly   (±10%)", [r.grand_total_match for r in results if not r.error]),
        ("Financeable total correct          (±10%)", [r.financeable_total_match for r in results if not r.error]),
        ("Subscription type correct",                [r.subscription_match for r in results if not r.error]),
        ("Subscription term correct",                [r.term_match for r in results if not r.error]),
        ("Vendor name found",                        [r.vendor_name_match for r in results if not r.error]),
        ("Customer name found",                      [r.customer_name_match for r in results if not r.error]),
        ("Quote ID found",                           [r.quote_id_match for r in results if not r.error]),
    ]

    for label, vals in checks:
        scored = [v for v in vals if v is not None]
        if not scored:
            continue
        passed = sum(1 for v in scored if v)
        pct = int(passed / len(scored) * 100)
        bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
        print(f"  {label:<42} {bar}  {passed}/{len(scored)}  ({pct}%)")

    # Classification scores
    print(f"\n  CLASSIFICATION ACCURACY (sample-based)")
    print(f"  {'─'*50}")
    for r in results:
        if r.error or r.classification_score is None:
            continue
        name = r.file.replace("_"," ").replace(".pdf","")[:37]
        pct = int(r.classification_score * 100)
        bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
        print(f"  {name:<38} {bar}  {pct}%")

    # Edge cases
    print(f"\n  EDGE CASE HANDLING")
    print(f"  {'─'*50}")
    for r in results:
        if r.error or not r.edge_case_flags:
            continue
        name = r.file.replace("_"," ").replace(".pdf","")[:30]
        print(f"  {name}")
        for label, passed in r.edge_case_flags:
            print(f"    {'✓' if passed else '✗'} {label}")

    # Latency detail
    print(f"\n  LATENCY vs 7s BUDGET")
    print(f"  {'─'*50}")
    for r in results:
        if r.error:
            continue
        name = r.file.replace("_"," ").replace(".pdf","")[:37]
        bar_len = min(int(r.latency_s), 14)
        budget_marker = 7
        bar = ""
        for i in range(14):
            if i < bar_len:
                bar += "█" if i < budget_marker else "▓"
            elif i == budget_marker - 1:
                bar += "|"
            else:
                bar += "░"
        status = "✓ within budget" if r.within_budget else f"✗ over by {r.latency_s - 7:.1f}s"
        print(f"  {name:<38} [{bar}] {r.latency_s:.1f}s  {status}")

    # Token & cost
    print(f"\n  TOKEN USAGE & COST  (model: {model})")
    print(f"  {'─'*50}")
    print(f"  {'Document':<38} {'Input':>8} {'Output':>8} {'Total':>8} {'Cost':>10}")
    for r in results:
        if r.error:
            continue
        name = r.file.replace("_"," ").replace(".pdf","")[:37]
        print(f"  {name:<38} {r.input_tokens:>8,} {r.output_tokens:>8,} {r.input_tokens+r.output_tokens:>8,} ${r.cost_usd:>8.4f}")
    print(f"  {'TOTAL':<38} {sum(r.input_tokens for r in results):>8,} {sum(r.output_tokens for r in results):>8,} {sum(r.input_tokens+r.output_tokens for r in results):>8,} ${total_cost:>8.4f}")

    # Confidence
    print(f"\n  CONFIDENCE SCORES (avg across 3 high-stakes fields)")
    print(f"  {'─'*50}")
    for r in results:
        if r.error or r.avg_confidence is None:
            continue
        name = r.file.replace("_"," ").replace(".pdf","")[:37]
        pct = int(r.avg_confidence * 100)
        bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
        print(f"  {name:<38} {bar}  {pct}%  ({r.flags_raised} flags)")

    # Overall score
    all_checks = []
    for r in results:
        if not r.error:
            for v in [r.grand_total_match, r.financeable_total_match, r.subscription_match,
                      r.vendor_name_match, r.customer_name_match]:
                if v is not None:
                    all_checks.append(v)

    overall_pct = int(sum(all_checks) / len(all_checks) * 100) if all_checks else 0
    target_met = "✓ TARGET MET" if overall_pct >= 90 else "✗ BELOW 90% TARGET"

    print(f"\n  {'='*50}")
    print(f"  OVERALL ACCURACY: {overall_pct}%   {target_met}")
    print(f"  LATENCY BUDGET:   {within_budget}/5 docs within 7s")
    print(f"  TOTAL COST:       ${total_cost:.4f} for {len(results)} extractions")
    print(f"  {'='*50}\n")

    return overall_pct

# ── Save JSON results ─────────────────────────────────────────────────────────

def save_results(results: list[DocResult], model: str, prompt_version: str, overall_pct: int):
    out = {
        "prompt_version": prompt_version,
        "model": model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "overall_accuracy_pct": overall_pct,
        "latency_budget_passed": sum(1 for r in results if r.within_budget),
        "total_cost_usd": round(sum(r.cost_usd for r in results), 4),
        "documents": []
    }
    for r in results:
        out["documents"].append({
            "file": r.file,
            "latency_s": round(r.latency_s, 2),
            "within_budget": r.within_budget,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "cost_usd": round(r.cost_usd, 4),
            "grand_total_match": r.grand_total_match,
            "financeable_total_match": r.financeable_total_match,
            "subscription_match": r.subscription_match,
            "vendor_name_match": r.vendor_name_match,
            "customer_name_match": r.customer_name_match,
            "classification_score": r.classification_score,
            "avg_confidence": r.avg_confidence,
            "flags_raised": r.flags_raised,
            "edge_case_flags": r.edge_case_flags,
            "error": r.error
        })

    out_path = Path(f"/home/claude/eval_results_{prompt_version}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Results saved to {out_path}")
    return out_path

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QuoteParser automated eval")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Anthropic model to use")
    parser.add_argument("--prompt", default=None, help="Path to prompt text file (default: built-in v1)")
    parser.add_argument("--version", default="v1", help="Prompt version label (e.g. v2)")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    if args.prompt:
        with open(args.prompt) as f:
            prompt = f.read()
        prompt_version = args.version
    else:
        prompt = DEFAULT_PROMPT
        prompt_version = args.version

    with open(GROUND_TRUTH_FILE) as f:
        ground_truth = json.load(f)

    gt_map = {doc["file"]: doc["ground_truth"] for doc in ground_truth["documents"]}

    results = []
    print(f"\nRunning eval — model: {args.model}, prompt: {prompt_version}")
    print(f"{'─'*50}")

    for doc in ground_truth["documents"]:
        fname = doc["file"]
        pdf_path = PDF_DIR / fname
        if not pdf_path.exists():
            # Try alternate naming
            alt = list(PDF_DIR.glob(f"*{fname.split('_')[0]}*"))
            if alt:
                pdf_path = alt[0]
                fname_display = pdf_path.name
            else:
                print(f"  SKIP  {fname} — file not found")
                r = DocResult(file=fname, error="file not found")
                results.append(r)
                continue
        else:
            fname_display = fname

        print(f"  → {fname_display[:55]}", end="", flush=True)
        result = DocResult(file=fname)

        try:
            pdf_b64 = load_pdf_b64(pdf_path)
            extracted, latency, input_tok, output_tok = extract(client, pdf_b64, prompt, args.model)

            result.latency_s = round(latency, 2)
            result.input_tokens = input_tok
            result.output_tokens = output_tok
            result.cost_usd = round((input_tok / 1_000_000 * COST_PER_1M_INPUT) + (output_tok / 1_000_000 * COST_PER_1M_OUTPUT), 4)
            result.extracted = extracted

            result = score_result(result, gt_map[fname])
            budget_str = "✓" if result.within_budget else f"✗ {result.latency_s:.1f}s"
            print(f"  {result.latency_s:.1f}s {budget_str}  {input_tok+output_tok:,} tokens  ${result.cost_usd:.4f}")

        except Exception as e:
            result.error = str(e)
            print(f"  ERROR: {e}")

        results.append(result)
        time.sleep(0.5)  # avoid rate limits

    overall_pct = print_report(results, args.model, prompt_version)
    save_results(results, args.model, prompt_version, overall_pct)

if __name__ == "__main__":
    main()
