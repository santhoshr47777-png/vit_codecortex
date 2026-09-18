import time
import threading
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


BASE = Path(__file__).resolve().parent
DATA_EMAIL = BASE / "data" / "email" / "emails.csv"
DATA_MALWARE = BASE / "data" / "malware" / "malware.csv"
MODELS = BASE / "models"


def train_email(progress=None):
    if progress:
        progress("EMAIL MODEL", "Loading emails.csv...", 10)

    df = pd.read_csv(DATA_EMAIL)
    X_text = df["text"].fillna("").astype(str)
    y = df["spam"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    if progress:
        progress("EMAIL MODEL", "Building TF-IDF features...", 25)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98
    )

    train_vec = vectorizer.fit_transform(X_train)
    test_vec = vectorizer.transform(X_test)

    if progress:
        progress("EMAIL MODEL", "Training Linear SVM...", 45)

    model = LinearSVC(
        class_weight="balanced",
        C=1.0,
        random_state=42
    )
    model.fit(train_vec, y_train)

    pred = model.predict(test_vec)

    scores = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0)
    }

    joblib.dump(model, MODELS / "email_model.joblib")
    joblib.dump(vectorizer, MODELS / "email_tfidf.joblib")

    if progress:
        progress("EMAIL MODEL", "Model saved successfully.", 55)

    return {
        "samples": len(df),
        "features": len(vectorizer.vocabulary_),
        "metrics": scores
    }


def train_malware(progress=None):
    if progress:
        progress("MALWARE MODEL", "Loading malware.csv...", 60)

    df = pd.read_csv(DATA_MALWARE, sep="|")

    if "legitimate" not in df.columns:
        raise ValueError("The malware dataset does not contain the 'legitimate' label.")

    y = (1 - df["legitimate"].astype(int))

    X = df.drop(columns=["legitimate"]).copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([float("inf"), float("-inf")], pd.NA).fillna(0)

    constant_columns = [
        c for c in X.columns
        if X[c].nunique(dropna=False) <= 1
    ]

    if constant_columns:
        X = X.drop(columns=constant_columns)

    features = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    if progress:
        progress(
            "MALWARE MODEL",
            f"Training Random Forest on {len(features)} features...",
            72
        )

    model = RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced",
        max_features="sqrt",
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    scores = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities)
    }

    joblib.dump(model, MODELS / "malware_model.joblib")
    joblib.dump(features, MODELS / "malware_features.joblib")

    if progress:
        progress("MALWARE MODEL", "Model saved successfully.", 92)

    return {
        "samples": len(df),
        "features": len(features),
        "metrics": scores
    }


def train_all(progress=None):
    started = time.time()

    email_result = train_email(progress)
    malware_result = train_malware(progress)

    elapsed = time.time() - started

    if progress:
        progress("COMPLETE", "Both models trained and saved.", 100)

    return {
        "email": email_result,
        "malware": malware_result,
        "seconds": elapsed
    }


if __name__ == "__main__":
    result = train_all(
        lambda stage, message, percent:
        print(f"[{percent:3d}%] {stage}: {message}")
    )

    print()
    print("TRAINING COMPLETE")
    print(result)
