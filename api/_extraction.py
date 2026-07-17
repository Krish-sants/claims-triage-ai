"""NLP extraction over the FNOL (First Notice of Loss) free-text description.

Default backend is a deterministic keyword/pattern extractor — transparent,
auditable, zero-latency. When ANTHROPIC_API_KEY is set (e.g. as a Vercel
environment variable) the extractor is upgraded to Claude for true
language understanding; the output schema is identical either way, which is
the orchestration pattern the rest of the pipeline relies on.
"""

import json
import os
import re
import urllib.request

INCIDENT_TYPES = {
    "collision": ["collision", "crash", "rear-end", "rear ended", "hit", "accident", "collided"],
    "theft": ["theft", "stolen", "burglary", "break-in", "broke in", "robbed"],
    "fire": ["fire", "burnt", "burned", "smoke damage", "flames"],
    "water_damage": ["flood", "water damage", "pipe burst", "leak", "leakage", "seepage"],
    "storm": ["storm", "cyclone", "hail", "wind damage", "tree fell"],
    "liability": ["slipped", "fell on", "third party injured", "customer injured"],
    "medical": ["hospitalised", "hospitalized", "surgery", "diagnosed", "treatment"],
}

FLAG_PATTERNS = {
    "injury_mentioned": ["injur", "whiplash", "fracture", "hospital", "pain", "medical attention", "ambulance"],
    "third_party_involved": ["third party", "other driver", "other vehicle", "pedestrian", "their car", "neighbour", "neighbor"],
    "attorney_mentioned": ["attorney", "lawyer", "solicitor", "legal notice", "advocate"],
    "police_report_filed": ["police", "fir", "first information report", "cops"],
    "total_loss_language": ["total loss", "totaled", "everything was", "lost everything", "all my belongings", "completely destroyed"],
    "cash_no_receipts": ["cash", "no receipt", "no bill", "lost the receipt", "no invoice", "no proof of purchase"],
}


# Negated mentions ("no injuries", "nobody was hurt") must not raise flags —
# the single most common failure of naive keyword extraction.
NEGATION_PATTERN = re.compile(
    r"\b(?:no|without|not|nobody|no[- ]one|none)\b[\w\s]{0,25}?"
    r"\b(?:injur\w*|hurt|hospital\w*|pain|attorney|lawyer|police)\b"
)


def _keyword_extract(description: str) -> dict:
    text = NEGATION_PATTERN.sub(" ", description.lower())
    scores = {
        incident: sum(1 for kw in keywords if kw in text)
        for incident, keywords in INCIDENT_TYPES.items()
    }
    best = max(scores, key=scores.get)
    incident_type = best if scores[best] > 0 else "other"
    flags = {
        flag: any(kw in text for kw in keywords)
        for flag, keywords in FLAG_PATTERNS.items()
    }
    amounts = re.findall(r"(?:rs\.?|inr|₹)\s?([\d,]+)", text)
    return {
        "incident_type": incident_type,
        **flags,
        "amounts_in_text": [a.replace(",", "") for a in amounts],
        "extraction_backend": "keyword-rules",
    }


def _claude_extract(description: str) -> dict:
    """LLM extraction — same schema, richer understanding. stdlib-only HTTP
    so the serverless bundle stays dependency-light."""
    schema_hint = {
        "incident_type": f"one of {list(INCIDENT_TYPES) + ['other']}",
        **{flag: "boolean" for flag in FLAG_PATTERNS},
        "amounts_in_text": "list of numeric strings",
    }
    body = json.dumps({
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 400,
        "system": "Extract claim facts from the FNOL description. "
                  f"Reply with ONLY a JSON object matching: {json.dumps(schema_hint)}",
        "messages": [{"role": "user", "content": description}],
    }).encode()
    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read())
    text = payload["content"][0]["text"]
    extracted = json.loads(re.search(r"\{.*\}", text, re.S).group())
    extracted["extraction_backend"] = "claude-api"
    return extracted


def extract_facts(description: str) -> dict:
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return _claude_extract(description)
        except Exception:
            pass  # graceful degradation — never fail triage because the LLM did
    return _keyword_extract(description)
