"""ClaimSense — AI Claims Triage API (FastAPI on Vercel serverless).

POST /api/triage : FNOL claim in -> extraction + severity + fraud + routing out.

Local dev:   uvicorn api.index:app --reload --port 8010
Deployed:    every /api/* request is rewritten to this ASGI app (vercel.json).
"""

import hashlib
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ._extraction import extract_facts
from ._fraud import score_fraud
from ._routing import draft_acknowledgement, route, score_severity

app = FastAPI(title="ClaimSense — AI Claims Triage", version="1.0.0")

# Vercel's FastAPI preset routes every request through this app, so the UI
# is served from here too instead of as a separate static file.
try:
    _INDEX_HTML = (Path(__file__).resolve().parent.parent / "index.html").read_text(encoding="utf-8")
except OSError:
    _INDEX_HTML = "<h1>ClaimSense</h1><p>UI not bundled — see /docs for the API.</p>"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return _INDEX_HTML


class ClaimIntake(BaseModel):
    claimant_name: str = Field(..., min_length=2, max_length=80)
    policy_type: str = Field(..., examples=["motor", "property", "health"])
    description: str = Field(..., min_length=20, max_length=5000,
                             description="FNOL free-text description of the incident")
    claim_amount: float = Field(..., gt=0)
    report_delay_days: int = Field(..., ge=0, le=365,
                                   description="Days between incident and report")
    policy_tenure_months: int = Field(..., ge=0, le=600)
    prior_claims_count: int = Field(..., ge=0, le=50)


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "claimsense-triage"}


@app.post("/api/triage")
def triage(claim: ClaimIntake):
    started = time.perf_counter()
    payload = claim.model_dump()

    facts = extract_facts(payload["description"])
    severity = score_severity(payload, facts)
    fraud = score_fraud(payload, facts)
    routing = route(severity, fraud)

    claim_ref = "CLM-" + hashlib.sha1(
        f"{payload['claimant_name']}{payload['description']}".encode()
    ).hexdigest()[:8].upper()

    return {
        "claim_ref": claim_ref,
        "extracted_facts": facts,
        "severity": severity,
        "fraud": fraud,
        "routing": routing,
        "customer_email_draft": draft_acknowledgement(payload, routing, claim_ref),
        "processing_ms": round((time.perf_counter() - started) * 1000, 1),
        "audit": {
            "pipeline": ["extract", "severity_model", "fraud_rules", "route", "draft_email"],
            "explainable": True,
            "pii_note": "No PII leaves the service; email draft is returned to the operator, not sent.",
        },
    }
