# QuoteParser — Eval Framework

## Setup
```bash
export ANTHROPIC_API_KEY=sk-ant-...
pip install anthropic
```

## Run eval (baseline)
```bash
python eval.py
```

## Test a new prompt version
1. Copy prompts/v1.txt to prompts/v2.txt
2. Edit prompts/v2.txt — change ONE thing
3. Run: python eval.py --prompt prompts/v2.txt --version v2
4. Compare output to v1 results in eval_results_v1.json

## Test a faster model (for latency)
```bash
python eval.py --model claude-haiku-4-5-20251001 --version v1-haiku
```

## What gets scored automatically
- Grand total extracted correctly (±10% tolerance)
- Financeable total correct (±10% tolerance)  
- Subscription type detected correctly
- Subscription term months correct
- Vendor name found
- Customer name found
- Quote ID found
- Classification accuracy (sample-based, checks key items)
- Edge case handling:
  - Dual price column detected (Apex)
  - Annualised EA pricing flagged (Citrus)
  - Labor excluded from financeable total (Network Refresh)
  - NRC vs recurring charges separated (Hartwell)

## Files
- eval.py              — main eval script
- ground_truth.json    — correct answers for all 5 docs
- prompts/v1.txt       — baseline prompt
- prompts/v2.txt       — your next iteration (create this)
- eval_results_v1.json — auto-generated after each run

## Prompt iteration workflow
1. Run v1, note what fails
2. Hypothesis: "I think X is failing because Y"
3. Change ONE thing in the prompt
4. Run v2, compare scores
5. If better: keep it. If worse: revert.
6. Repeat until 90%+ accuracy and <7s latency

## Known edge cases to watch
- Apex: MSRP vs discounted price — agent must use discounted column
- Citrus: Annualised EA pricing — $143k shown is per-year, not total contract
- Network Refresh: Labor ($7,500) must be excluded from financeable total
- Hartwell: NRC ($45,300) separate from recurring ACV ($552,012)
- RPA: 38 line items, 8 pages — latency risk, duplication risk
