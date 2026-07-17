# ClaimSense — AI Claims Triage & Fraud Intelligence

GenAI-assisted First Notice of Loss (FNOL) triage for insurers: one API call
takes a raw claim (free-text description + structured fields) and returns
extracted facts, a severity prediction, explainable fraud red flags, a
routing decision with SLA, and a ready-to-send customer acknowledgement.

**The business problem:** claims triage is the most expensive bottleneck in
P&C insurance. Manual FNOL review takes hours-to-days, low-value clean
claims clog adjuster queues, and ~10% of claim spend leaks to fraud that is
cheapest to catch at intake. ClaimSense triages in ~5 ms per claim: clean
low-severity claims route to same-day auto-settlement, injury/attorney
claims go straight to senior adjusters, and high-red-flag claims are
referred to SIU with a written, auditable reason for every point of score.

## Live demo

**https://claims-triage-ai.vercel.app** — deployed on Vercel serverless.
Try the two sample buttons (clean motor claim vs suspicious theft claim).

## Architecture

```
FNOL claim ──► FastAPI (Pydantic validation)
                 │
                 ├─ 1. EXTRACTION  free text -> structured facts
                 │     keyword+negation rules by default; auto-upgrades to
                 │     Claude when ANTHROPIC_API_KEY is set (same schema)
                 │
                 ├─ 2. SEVERITY    Logistic Regression (ROC-AUC 0.92),
                 │     trained offline with sklearn, served as pure-Python
                 │     coefficients (api/_model.py) — no ML libs at runtime
                 │
                 ├─ 3. FRAUD       weighted, named red-flag rules ->
                 │     score + tier + SIU referral, every flag explained
                 │
                 └─ 4. ROUTING     severity × fraud -> queue + SLA
                       + drafted customer acknowledgement email
```

### Design decisions (interview material)

- **Train-heavy, serve-light.** sklearn trains the severity model; the
  artefact is exported as plain coefficients into `api/_model.py`. Inference
  is a dot product + sigmoid — zero heavy deps, tiny cold start, fits any
  serverless/edge platform, and the model change shows up in code review.
- **LLM orchestration with graceful degradation.** The extractor interface
  is backend-agnostic: deterministic rules by default, Claude API
  when a key is configured, and an automatic fallback if the LLM call fails.
  Triage must never go down because a model provider did.
- **Negation-aware NLP.** "No injuries" must not raise the injury flag —
  handled explicitly (see `NEGATION_PATTERN`), and regression-tested.
- **Responsible AI = explainability + PII discipline.** Every fraud point is
  a named rule with a weight and a human-readable reason (defensible to a
  regulator); the email draft is returned to the operator, never auto-sent.

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\pip install fastapi "uvicorn[standard]" pytest httpx
.venv\Scripts\python -m uvicorn api.index:app --reload --port 8010
# open http://127.0.0.1:8010/docs, or serve index.html for the UI
.venv\Scripts\python -m pytest tests -q
```

Retrain the severity model (needs sklearn/pandas):
`python training/train_severity.py` — regenerates `api/_model.py`.

## Deploy

`vercel --prod` from the repo root. `vercel.json` rewrites `/api/*` to the
FastAPI app; `index.html` is served statically. Optionally set
`ANTHROPIC_API_KEY` in Vercel project settings to enable LLM extraction.
