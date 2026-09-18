"""
ThreatLens X
Novel Threat Detection

Pipeline:
Email text
    ↓
TF-IDF
    ↓
Truncated SVD
    ↓
Isolation Forest
    ↓
Anomaly / Novelty Score
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMAIL_FILE = (
    PROJECT_ROOT /
    "data" /
    "email" /
    "emails.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"


# ============================================================
# LOAD
# ============================================================

def load_email_data():

    df = pd.read_csv(
        EMAIL_FILE,
        low_memory=False
    )

    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
    )

    return df


def load_vectorizer():

    return joblib.load(
        MODEL_DIR /
        "email_tfidf.joblib"
    )


# ============================================================
# TRAIN NOVELTY MODEL
# ============================================================

def train_novelty_model():

    print("=" * 70)
    print("        THREATLENS X - NOVEL THREAT DETECTOR")
    print("=" * 70)

    df = load_email_data()

    vectorizer = load_vectorizer()

    print(
        f"\nEmail samples: {len(df):,}"
    )

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    print("\nCreating TF-IDF representation...")

    X_tfidf = vectorizer.transform(
        df["text"]
    )

    print(
        f"TF-IDF shape: {X_tfidf.shape}"
    )

    # --------------------------------------------------------
    # Dimensionality reduction
    # --------------------------------------------------------

    print("\nReducing feature space with SVD...")

    svd = TruncatedSVD(
        n_components=100,
        random_state=42
    )

    X_reduced = svd.fit_transform(
        X_tfidf
    )

    print(
        f"SVD shape: {X_reduced.shape}"
    )

    print(
        f"Explained variance: "
        f"{svd.explained_variance_ratio_.sum():.4f}"
    )

    # --------------------------------------------------------
    # Isolation Forest
    # --------------------------------------------------------

    print("\nTraining Isolation Forest...")

    detector = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )

    detector.fit(
        X_reduced
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    joblib.dump(
        svd,
        MODEL_DIR /
        "email_svd.joblib"
    )

    joblib.dump(
        detector,
        MODEL_DIR /
        "email_novelty_detector.joblib"
    )

    print("\nSaved:")

    print(
        "  models/email_svd.joblib"
    )

    print(
        "  models/email_novelty_detector.joblib"
    )

    return detector, svd


# ============================================================
# SCORE NEW EMAIL
# ============================================================

def novelty_score(text):

    vectorizer = load_vectorizer()

    svd = joblib.load(
        MODEL_DIR /
        "email_svd.joblib"
    )

    detector = joblib.load(
        MODEL_DIR /
        "email_novelty_detector.joblib"
    )

    vector = vectorizer.transform(
        [text]
    )

    reduced = svd.transform(
        vector
    )

    # Isolation Forest:
    # higher decision_function = more normal
    # lower decision_function = more anomalous

    raw_score = detector.decision_function(
        reduced
    )[0]

    prediction = detector.predict(
        reduced
    )[0]

    # Convert raw score into an easier-to-display
    # novelty score.
    #
    # This is a relative model score, NOT a probability.

    novelty = 50 - (raw_score * 100)

    novelty = max(
        0,
        min(
            100,
            novelty
        )
    )

    if prediction == -1:
        status = "NOVEL / ANOMALOUS"
    else:
        status = "WITHIN LEARNED PATTERN"

    return {
        "raw_score": float(raw_score),
        "novelty_score": round(
            float(novelty),
            2
        ),
        "status": status
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    train_novelty_model()

    print("\n" + "=" * 70)
    print("              NOVELTY TEST")
    print("=" * 70)

    sample = """
    Urgent security notification.
    Your account requires immediate verification.
    Please confirm your password using the provided link.
    """

    result = novelty_score(
        sample
    )

    print(
        f"\nRaw anomaly score : "
        f"{result['raw_score']:.6f}"
    )

    print(
        f"Novelty score     : "
        f"{result['novelty_score']}/100"
    )

    print(
        f"Status            : "
        f"{result['status']}"
    )