"""
ThreatLens X
Risk Scoring Engine
"""

from dataclasses import dataclass


@dataclass
class RiskResult:
    score: float
    severity: str
    explanation: str


def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(value, maximum))


def calculate_risk(
    model_confidence,
    anomaly_score=0,
    similarity_score=0,
    evidence_score=0
):
    """
    Calculate a normalized ThreatLens risk score.

    Inputs:
        model_confidence : 0-100
        anomaly_score    : 0-100
        similarity_score : 0-100
        evidence_score   : 0-100

    Returns:
        RiskResult
    """

    model_confidence = clamp(model_confidence)
    anomaly_score = clamp(anomaly_score)
    similarity_score = clamp(similarity_score)
    evidence_score = clamp(evidence_score)

    # Weighted risk model
    #
    # Model confidence is the strongest signal.
    # The other signals will become more meaningful
    # when we implement anomaly/similarity engines.

    score = (
        model_confidence * 0.50
        + evidence_score * 0.20
        + similarity_score * 0.15
        + anomaly_score * 0.15
    )

    score = round(
        clamp(score),
        2
    )

    # Severity
    if score >= 85:
        severity = "CRITICAL"

    elif score >= 70:
        severity = "HIGH"

    elif score >= 40:
        severity = "MEDIUM"

    else:
        severity = "LOW"

    explanation = (
        f"Risk score {score}/100. "
        f"Severity classified as {severity}."
    )

    return RiskResult(
        score=score,
        severity=severity,
        explanation=explanation
    )


# ------------------------------------------------------------
# Quick test
# ------------------------------------------------------------

if __name__ == "__main__":

    result = calculate_risk(
        model_confidence=92,
        anomaly_score=70,
        similarity_score=88,
        evidence_score=90
    )

    print("=" * 50)
    print("THREATLENS RISK ENGINE TEST")
    print("=" * 50)

    print(f"Risk     : {result.score}/100")
    print(f"Severity : {result.severity}")
    print(f"Message  : {result.explanation}")