from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


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
# LOAD DATA + VECTORIZER
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
# THREAT DNA
# ============================================================

def create_threat_dna(text):

    vectorizer = load_vectorizer()

    vector = vectorizer.transform(
        [text]
    )

    feature_names = (
        vectorizer
        .get_feature_names_out()
    )

    values = vector.toarray()[0]

    # Get strongest TF-IDF features
    indices = values.argsort()[::-1]

    dna = []

    for index in indices:

        if values[index] <= 0:
            continue

        dna.append({
            "feature": feature_names[index],
            "strength": round(
                float(values[index]),
                4
            )
        })

        if len(dna) >= 15:
            break

    return dna


# ============================================================
# SIMILARITY SEARCH
# ============================================================

def find_similar_emails(
    new_text,
    top_k=5
):

    df = load_email_data()
    vectorizer = load_vectorizer()

    # Historical email vectors
    historical_vectors = vectorizer.transform(
        df["text"]
    )

    # New email vector
    new_vector = vectorizer.transform(
        [new_text]
    )

    # Cosine similarity
    similarities = cosine_similarity(
        new_vector,
        historical_vectors
    )[0]

    # Highest similarity first
    top_indices = similarities.argsort()[
        ::-1
    ][:top_k]

    results = []

    for index in top_indices:

        results.append({
            "dataset_index": int(index),
            "similarity": round(
                float(similarities[index]) * 100,
                2
            ),
            "label": int(
                df.iloc[index]["spam"]
            ),
            "text": df.iloc[index]["text"][
                :300
            ]
        })

    return results


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    sample_email = """
    Urgent! Your account has been suspended.
    Click the link immediately to verify your
    password and account information.
    """

    print("=" * 65)
    print("              THREAT DNA")
    print("=" * 65)

    dna = create_threat_dna(
        sample_email
    )

    for item in dna:

        print(
            f"{item['feature']:30}"
            f"{item['strength']:.4f}"
        )

    print("\n" + "=" * 65)
    print("           SIMILAR HISTORICAL EMAILS")
    print("=" * 65)

    results = find_similar_emails(
        sample_email,
        top_k=5
    )

    for result in results:

        label = (
            "SPAM"
            if result["label"] == 1
            else "LEGITIMATE"
        )

        print(
            f"\nIndex      : "
            f"{result['dataset_index']}"
        )

        print(
            f"Similarity : "
            f"{result['similarity']}%"
        )

        print(
            f"Label      : {label}"
        )

        print(
            f"Text       : "
            f"{result['text']}"
        )