"""
ThreatLens X
Explainable Email Detection
"""

from pathlib import Path

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT /
    "models" /
    "email_model.joblib"
)

VECTORIZER_PATH = (
    PROJECT_ROOT /
    "models" /
    "email_tfidf.joblib"
)


def load_models():

    model = joblib.load(
        MODEL_PATH
    )

    vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    return model, vectorizer


def explain_email(text, top_n=10):

    model, vectorizer = load_models()

    # Transform email
    vector = vectorizer.transform(
        [text]
    )

    # Prediction
    prediction = model.predict(
        vector
    )[0]

    decision_score = model.decision_function(
        vector
    )[0]

    # Feature names
    feature_names = np.array(
        vectorizer.get_feature_names_out()
    )

    # Values present in this email
    row = vector.toarray()[0]

    nonzero = np.where(
        row != 0
    )[0]

    # Linear SVM coefficients
    coefficients = model.coef_[0]

    contributions = []

    for index in nonzero:

        contribution = (
            row[index] *
            coefficients[index]
        )

        contributions.append(
            (
                feature_names[index],
                float(contribution)
            )
        )

    # Sort strongest absolute contributions
    contributions.sort(
        key=lambda x: abs(x[1]),
        reverse=True
    )

    top_features = contributions[:top_n]

    return {
        "prediction": int(prediction),
        "decision_score": float(decision_score),
        "features": top_features
    }


def print_explanation(result):

    print("\n" + "=" * 60)
    print("              EMAIL EXPLANATION")
    print("=" * 60)

    if result["prediction"] == 1:
        print("\nPrediction: SPAM")
    else:
        print("\nPrediction: LEGITIMATE")

    print(
        f"Decision score: "
        f"{result['decision_score']:.4f}"
    )

    print("\nImportant contributing features:")

    for feature, contribution in result["features"]:

        direction = (
            "SPAM"
            if contribution > 0
            else "LEGITIMATE"
        )

        print(
            f"  {feature:30} "
            f"{contribution:+.4f} "
            f"→ {direction}"
        )


# ------------------------------------------------------------
# Test
# ------------------------------------------------------------

if __name__ == "__main__":

    sample_email = """
    Urgent! Your account has been suspended.
    Click here immediately to verify your account
    and confirm your password.
    """

    result = explain_email(
        sample_email
    )

    print_explanation(
        result
    )