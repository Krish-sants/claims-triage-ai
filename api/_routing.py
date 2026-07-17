"""Severity scoring (pure-Python inference) + routing decision + customer email.

The severity model is the artefact exported by training/train_severity.py;
inference is a dot product and a sigmoid — no ML libraries in the runtime.
Routing crosses severity with the fraud tier to pick a queue and SLA, which
is where triage becomes money: low-severity clean claims settle same-day
instead of waiting in an adjuster's queue.
"""

import math

from ._model import MODEL

SEVERITY_TIERS = [(0.25, "low"), (0.55, "medium"), (0.85, "high"), (1.01, "catastrophic")]

ROUTES = {
    # (severity_tier, siu_referral) -> queue, SLA hours
    ("low", False): ("auto_settlement", 4),
    ("medium", False): ("fast_track_adjuster", 24),
    ("high", False): ("senior_adjuster", 48),
    ("catastrophic", False): ("major_loss_team", 24),
}


def score_severity(claim: dict, facts: dict) -> dict:
    values = {
        "log_claim_amount": math.log(max(claim["claim_amount"], 1)),
        "report_delay_days": claim["report_delay_days"],
        "policy_tenure_months": claim["policy_tenure_months"],
        "prior_claims_count": claim["prior_claims_count"],
        "injury_mentioned": int(bool(facts.get("injury_mentioned"))),
        "third_party_involved": int(bool(facts.get("third_party_involved"))),
        "attorney_mentioned": int(bool(facts.get("attorney_mentioned"))),
        "incident_is_liability_or_medical": int(facts.get("incident_type") in ("liability", "medical")),
    }
    z = MODEL["intercept"]
    for feature, mean, std, coef in zip(MODEL["features"], MODEL["means"], MODEL["stds"], MODEL["coefficients"]):
        z += coef * ((values[feature] - mean) / std)
    probability = 1 / (1 + math.exp(-z))
    tier = next(label for cutoff, label in SEVERITY_TIERS if probability < cutoff)
    return {"severity_score": round(probability, 4), "severity_tier": tier,
            "model_roc_auc": MODEL["test_roc_auc"]}


def route(severity: dict, fraud: dict) -> dict:
    if fraud["siu_referral"]:
        queue, sla = "siu_investigation", 72
    else:
        queue, sla = ROUTES[(severity["severity_tier"], False)]
    return {"queue": queue, "sla_hours": sla}


def draft_acknowledgement(claim: dict, routing: dict, claim_ref: str) -> str:
    next_step = {
        "auto_settlement": "Your claim qualifies for accelerated settlement. Expect an update within 4 business hours.",
        "fast_track_adjuster": "A claims specialist will contact you within 1 business day.",
        "senior_adjuster": "A senior adjuster has been assigned and will contact you within 2 business days.",
        "major_loss_team": "Our major loss team has been engaged and will contact you within 1 business day.",
        "siu_investigation": "Your claim is under standard review; our team will contact you within 3 business days.",
    }[routing["queue"]]
    return (
        f"Dear {claim['claimant_name']},\n\n"
        f"We have received your {claim['policy_type']} claim and registered it under reference {claim_ref}. "
        f"{next_step}\n\n"
        "You can reply to this email to add documents or information at any time.\n\n"
        "Kind regards,\nClaims Team"
    )
