from fastapi.testclient import TestClient

from api.index import app

client = TestClient(app)

CLEAN_CLAIM = {
    "claimant_name": "Asha Verma",
    "policy_type": "motor",
    "description": "Another car rear ended me at a signal. Minor bumper damage. "
                   "Police report filed at the scene, other driver accepted fault. No injuries.",
    "claim_amount": 45000,
    "report_delay_days": 1,
    "policy_tenure_months": 26,
    "prior_claims_count": 0,
}

SUSPICIOUS_CLAIM = {
    "claimant_name": "Test User",
    "policy_type": "property",
    "description": "My house was broken into. Everything was stolen, all my belongings "
                   "completely gone. Items were bought with cash so I have no receipts. "
                   "My lawyer said to claim the full amount.",
    "claim_amount": 500000,
    "report_delay_days": 21,
    "policy_tenure_months": 2,
    "prior_claims_count": 2,
}


def test_health():
    assert client.get("/api/health").json()["status"] == "ok"


def test_clean_claim_fast_tracked():
    result = client.post("/api/triage", json=CLEAN_CLAIM).json()
    assert result["extracted_facts"]["incident_type"] == "collision"
    assert result["fraud"]["fraud_tier"] == "low"
    assert not result["fraud"]["siu_referral"]
    assert result["routing"]["queue"] in ("auto_settlement", "fast_track_adjuster")


def test_suspicious_claim_referred_to_siu():
    result = client.post("/api/triage", json=SUSPICIOUS_CLAIM).json()
    assert result["extracted_facts"]["incident_type"] == "theft"
    assert result["fraud"]["fraud_tier"] == "high"
    assert result["fraud"]["siu_referral"]
    assert result["routing"]["queue"] == "siu_investigation"
    assert len(result["fraud"]["red_flags"]) >= 4


def test_validation_rejects_bad_payload():
    bad = dict(CLEAN_CLAIM, claim_amount=-5)
    assert client.post("/api/triage", json=bad).status_code == 422
