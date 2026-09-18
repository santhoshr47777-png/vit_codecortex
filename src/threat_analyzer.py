"""
ThreatLens X
Unified Threat Analyzer

Combines:

1. Email classification
2. Threat DNA
3. Historical similarity
4. Novelty detection
5. Risk scoring
"""

from pathlib import Path
import sys

import joblib
import numpy as np


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from similarity import (
    create_threat_dna,
    find_similar_emails
)

from novelty_detector import (
    novelty_score
)

from risk_engine import (
    calculate_risk
)


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_DIR = PROJECT_ROOT / "models"

EMAIL_MODEL = (
    MODEL_DIR /
    "email_model.joblib"
)

EMAIL_VECTORIZER = (
    MODEL_DIR /
    "email_tfidf.joblib"
)


# ============================================================
# LOAD EMAIL MODEL
# ============================================================

def load_email_model():

    model = joblib.load(
        EMAIL_MODEL
    )

    vectorizer = joblib.load(
        EMAIL_VECTORIZER
    )

    return model, vectorizer


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_email(text):

    model, vectorizer = (
        load_email_model()
    )

    vector = vectorizer.transform(
        [text]
    )

    prediction = model.predict(
        vector
    )[0]

    decision_score = model.decision_function(
        vector
    )[0]

    # Linear SVM does not produce a true probability.
    # We convert the magnitude into a bounded confidence
    # score for dashboard display.
    #
    # This is a confidence-like score, not a calibrated
    # probability.

    confidence = (
        abs(decision_score) /
        (1 + abs(decision_score))
    ) * 100

    confidence = round(
        float(confidence),
        2
    )

    label = (
        "SPAM"
        if prediction == 1
        else "LEGITIMATE"
    )

    return {
        "prediction": int(prediction),
        "label": label,
        "decision_score": float(
            decision_score
        ),
        "confidence": confidence
    }


# ============================================================
# EVIDENCE SCORE
# ============================================================

def calculate_evidence_score(
    threat_dna
):

    if not threat_dna:

        return 0.0

    strengths = [
        item["strength"]
        for item in threat_dna
    ]

    average = (
        sum(strengths) /
        len(strengths)
    )

    # Convert relative TF-IDF strength into
    # a bounded evidence score.

    score = min(
        100,
        average * 200
    )

    return round(
        score,
        2
    )


# ============================================================
# COMPLETE ANALYSIS
# ============================================================

def analyze_email(text):

    # --------------------------------------------------------
    # 1. ML CLASSIFICATION
    # --------------------------------------------------------

    classification = classify_email(
        text
    )

    # --------------------------------------------------------
    # 2. THREAT DNA
    # --------------------------------------------------------

    dna = create_threat_dna(
        text
    )

    evidence_score = (
        calculate_evidence_score(
            dna
        )
    )

    # --------------------------------------------------------
    # 3. SIMILARITY
    # --------------------------------------------------------

    similar = find_similar_emails(
        text,
        top_k=5
    )

    if similar:

        similarity_score = (
            similar[0]["similarity"]
        )

    else:

        similarity_score = 0

    # --------------------------------------------------------
    # 4. NOVELTY
    # --------------------------------------------------------

    novelty = novelty_score(
        text
    )

    novelty_value = (
        novelty["novelty_score"]
    )

    # --------------------------------------------------------
    # 5. RISK
    # --------------------------------------------------------

    # Only malicious/spam classifications
    # should receive the strongest model-risk signal.

    if classification["prediction"] == 1:

        model_confidence = (
            classification["confidence"]
        )

    else:

        model_confidence = max(
            0,
            100 -
            classification["confidence"]
        )

    risk = calculate_risk(
        model_confidence=model_confidence,
        anomaly_score=novelty_value,
        similarity_score=similarity_score,
        evidence_score=evidence_score
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    return {

        "classification": classification,

        "threat_dna": dna,

        "similar_threats": similar,

        "novelty": novelty,

        "evidence_score": evidence_score,

        "similarity_score": similarity_score,

        "risk": {
            "score": risk.score,
            "severity": risk.severity,
            "explanation": risk.explanation
        }
    }


# ============================================================
# PRINT REPORT
# ============================================================

def print_report(result):

    classification = (
        result["classification"]
    )

    risk = result["risk"]

    print("\n")
    print("=" * 70)
    print("                 THREATLENS X")
    print("              THREAT INVESTIGATION")
    print("=" * 70)

    print("\n[CLASSIFICATION]")

    print(
        f"Prediction : "
        f"{classification['label']}"
    )

    print(
        f"Confidence : "
        f"{classification['confidence']:.2f}"
    )

    print(
        f"Decision   : "
        f"{classification['decision_score']:.4f}"
    )

    print("\n[THREAT DNA]")

    if result["threat_dna"]:

        for item in result["threat_dna"]:

            print(
                f"  {item['feature']:30}"
                f"{item['strength']:.4f}"
            )

    else:

        print("  No significant features found.")

    print("\n[SIMILAR HISTORICAL THREATS]")

    if result["similar_threats"]:

        for item in result["similar_threats"]:

            label = (
                "SPAM"
                if item["label"] == 1
                else "LEGITIMATE"
            )

            print(
                f"  #{item['dataset_index']:5} | "
                f"{item['similarity']:6.2f}% | "
                f"{label}"
            )

    else:

        print("  No similar samples found.")

    print("\n[NOVELTY ANALYSIS]")

    print(
        f"Novelty score : "
        f"{result['novelty']['novelty_score']:.2f}/100"
    )

    print(
        f"Status        : "
        f"{result['novelty']['status']}"
    )

    print("\n[RISK ANALYSIS]")

    print(
        f"Model signal  : "
        f"{classification['confidence']:.2f}"
    )

    print(
        f"Evidence      : "
        f"{result['evidence_score']:.2f}"
    )

    print(
        f"Similarity    : "
        f"{result['similarity_score']:.2f}"
    )

    print(
        f"Novelty       : "
        f"{result['novelty']['novelty_score']:.2f}"
    )

    print(
        f"\nFINAL RISK    : "
        f"{risk['score']:.2f}/100"
    )

    print(
        f"SEVERITY      : "
        f"{risk['severity']}"
    )

    print(
        f"\n{risk['explanation']}"
    )

    print("\n" + "=" * 70)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    sample_email = """
    URGENT NOTICE!

    Your account has been suspended.

    You must verify your account immediately
    by confirming your password and financial
    information using the link below.

    Failure to verify your account within 24 hours
    will result in permanent account suspension.
    """

    result = analyze_email(
        sample_email
    )

    print_report(
        result
    )