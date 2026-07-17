"""Fraud red-flag engine — weighted, explainable rules.

Insurers lose ~10% of claim spend to fraud. Black-box fraud scores are
hard to act on and harder to defend; every point of this score is a named
rule with a weight, so an investigator (and a regulator) can see exactly
why a claim was referred. This transparency IS the Responsible AI feature.
"""

from dataclasses import dataclass


@dataclass
class RedFlag:
    rule: str
    weight: float
    detail: str


def score_fraud(claim: dict, facts: dict) -> dict:
    flags: list[RedFlag] = []

    if claim["report_delay_days"] > 14:
        flags.append(RedFlag("late_reporting", 0.18,
                             f"Reported {claim['report_delay_days']} days after incident (threshold 14)."))
    if claim["policy_tenure_months"] < 3:
        flags.append(RedFlag("early_tenure_claim", 0.22,
                             f"Claim within {claim['policy_tenure_months']} month(s) of policy inception."))
    if claim["claim_amount"] >= 5000 and claim["claim_amount"] % 1000 == 0:
        flags.append(RedFlag("round_amount", 0.08,
                             f"Suspiciously round claim amount ({claim['claim_amount']:,})."))
    if claim["prior_claims_count"] >= 2:
        flags.append(RedFlag("claim_frequency", 0.16,
                             f"{claim['prior_claims_count']} prior claims in the last 3 years."))
    if facts.get("attorney_mentioned") and not facts.get("police_report_filed"):
        flags.append(RedFlag("attorney_no_police_report", 0.15,
                             "Attorney involved at FNOL but no police report mentioned."))
    if facts.get("total_loss_language"):
        flags.append(RedFlag("total_loss_language", 0.10,
                             "Description uses total-loss language ('everything', 'completely destroyed')."))
    if facts.get("cash_no_receipts"):
        flags.append(RedFlag("undocumented_items", 0.12,
                             "Cash purchases / missing receipts mentioned."))
    if facts.get("injury_mentioned") and claim["claim_amount"] < 20000:
        flags.append(RedFlag("injury_low_damage", 0.10,
                             "Injury claimed on a low-value loss — classic soft-fraud pattern."))

    score = min(1.0, round(sum(f.weight for f in flags), 3))
    tier = "high" if score >= 0.45 else "medium" if score >= 0.25 else "low"
    return {
        "fraud_score": score,
        "fraud_tier": tier,
        "red_flags": [vars(f) for f in flags],
        "siu_referral": tier == "high",
    }
