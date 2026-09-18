from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMAIL_FILE = PROJECT_ROOT / "data" / "email" / "emails.csv"
MALWARE_FILE = PROJECT_ROOT / "data" / "malware" / "malware.csv"

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


# ============================================================
# UTILITY
# ============================================================

def calculate_metrics(y_true, y_pred, scores=None):

    result = {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "confusion_matrix": (
            confusion_matrix(y_true, y_pred)
            .tolist()
        )
    }

    if scores is not None:

        try:

            result["roc_auc"] = float(
                roc_auc_score(y_true, scores)
            )

        except Exception:
            pass

    return result


# ============================================================
# EMAIL MODEL
# ============================================================

def train_email_model():

    print("\n" + "=" * 70)
    print("                 EMAIL MODEL TRAINING")
    print("=" * 70)

    df = pd.read_csv(
        EMAIL_FILE,
        low_memory=False
    )

    print(f"\nOriginal samples: {len(df):,}")

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    required_columns = {"text", "spam"}

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing email columns: {missing}"
        )

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    df = df[["text", "spam"]].copy()

    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["spam"] = pd.to_numeric(
        df["spam"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["spam"]
    )

    df = df[
        df["text"].str.len() > 0
    ]

    # Remove duplicate emails
    before = len(df)

    df = df.drop_duplicates(
        subset=["text"]
    )

    print(
        f"Duplicates removed: "
        f"{before - len(df):,}"
    )

    print(
        f"Final samples: {len(df):,}"
    )

    print("\nClass distribution:")
    print(df["spam"].value_counts())

    X = df["text"]
    y = df["spam"].astype(int)

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(
        f"\nTraining samples: {len(X_train):,}"
    )

    print(
        f"Testing samples : {len(X_test):,}"
    )

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    print("\nBuilding TF-IDF features...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98
    )

    X_train_tfidf = vectorizer.fit_transform(
        X_train
    )

    X_test_tfidf = vectorizer.transform(
        X_test
    )

    print(
        f"TF-IDF features: "
        f"{X_train_tfidf.shape[1]:,}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nTraining Linear SVM...")

    start = time.time()

    model = LinearSVC(
        class_weight="balanced",
        C=1.0,
        random_state=42
    )

    model.fit(
        X_train_tfidf,
        y_train
    )

    training_time = time.time() - start

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    predictions = model.predict(
        X_test_tfidf
    )

    decision_scores = model.decision_function(
        X_test_tfidf
    )

    metrics = calculate_metrics(
        y_test,
        predictions,
        decision_scores
    )

    print("\nEMAIL MODEL RESULTS")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Legitimate",
                "Spam"
            ],
            zero_division=0
        )
    )

    print(
        f"Training time: "
        f"{training_time:.2f} seconds"
    )

    print(
        f"Accuracy : {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {metrics['precision']:.4f}"
    )

    print(
        f"Recall   : {metrics['recall']:.4f}"
    )

    print(
        f"F1       : {metrics['f1']:.4f}"
    )

    if "roc_auc" in metrics:

        print(
            f"ROC-AUC  : "
            f"{metrics['roc_auc']:.4f}"
        )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    joblib.dump(
        model,
        MODEL_DIR / "email_model.joblib"
    )

    joblib.dump(
        vectorizer,
        MODEL_DIR / "email_tfidf.joblib"
    )

    print("\nSaved:")
    print("  models/email_model.joblib")
    print("  models/email_tfidf.joblib")

    return {
        "dataset": str(EMAIL_FILE),
        "samples": len(df),
        "features": X_train_tfidf.shape[1],
        "training_time_seconds": training_time,
        "metrics": metrics
    }


# ============================================================
# MALWARE MODEL
# ============================================================

def train_malware_model():

    print("\n" + "=" * 70)
    print("                MALWARE MODEL TRAINING")
    print("=" * 70)

    # Malware CSV is pipe-separated
    df = pd.read_csv(
        MALWARE_FILE,
        sep="|",
        low_memory=False
    )

    print(
        f"\nOriginal samples: {len(df):,}"
    )

    if "legitimate" not in df.columns:

        raise ValueError(
            "Malware label column "
            "'legitimate' was not found."
        )

    # --------------------------------------------------------
    # Remove useless / non-feature columns
    # --------------------------------------------------------

    # Keep the target separate
    y_original = pd.to_numeric(
        df["legitimate"],
        errors="coerce"
    )

    # Convert:
    # legitimate = 1 -> threat = 0
    # legitimate = 0 -> threat = 1
    #
    # This gives ThreatLens a common convention:
    # 0 = benign
    # 1 = malicious

    y = (1 - y_original)

    valid = y.notna()

    df = df.loc[valid].copy()
    y = y.loc[valid].astype(int)

    # --------------------------------------------------------
    # Select numeric features
    # --------------------------------------------------------

    X = df.drop(
        columns=["legitimate"],
        errors="ignore"
    )

    X = X.select_dtypes(
        include=np.number
    )

    # Replace invalid numerical values
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(0)

    # Remove constant columns
    nunique = X.nunique()

    constant_columns = nunique[
        nunique <= 1
    ].index

    if len(constant_columns) > 0:

        X = X.drop(
            columns=constant_columns
        )

    print(
        f"Numeric features: {X.shape[1]:,}"
    )

    print("\nNormalized class distribution:")

    print(
        y.value_counts()
        .sort_index()
        .rename(
            index={
                0: "Benign",
                1: "Malicious"
            }
        )
    )

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(
        f"\nTraining samples: {len(X_train):,}"
    )

    print(
        f"Testing samples : {len(X_test):,}"
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    print("\nTraining Random Forest...")

    start = time.time()

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    training_time = time.time() - start

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    metrics = calculate_metrics(
        y_test,
        predictions,
        probabilities
    )

    print("\nMALWARE MODEL RESULTS")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Benign",
                "Malicious"
            ],
            zero_division=0
        )
    )

    print(
        f"Training time: "
        f"{training_time:.2f} seconds"
    )

    print(
        f"Accuracy : {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {metrics['precision']:.4f}"
    )

    print(
        f"Recall   : {metrics['recall']:.4f}"
    )

    print(
        f"F1       : {metrics['f1']:.4f}"
    )

    if "roc_auc" in metrics:

        print(
            f"ROC-AUC  : "
            f"{metrics['roc_auc']:.4f}"
        )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importances = pd.Series(
        model.feature_importances_,
        index=X.columns
    )

    importances = (
        importances
        .sort_values(ascending=False)
        .head(20)
    )

    print("\nTOP MALWARE FEATURES")

    for feature, importance in importances.items():

        print(
            f"{feature:35} "
            f"{importance:.6f}"
        )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    joblib.dump(
        model,
        MODEL_DIR / "malware_model.joblib"
    )

    joblib.dump(
        list(X.columns),
        MODEL_DIR / "malware_features.joblib"
    )

    print("\nSaved:")
    print("  models/malware_model.joblib")
    print("  models/malware_features.joblib")

    return {
        "dataset": str(MALWARE_FILE),
        "samples": len(df),
        "features": X.shape[1],
        "training_time_seconds": training_time,
        "metrics": metrics,
        "top_features": {
            str(feature): float(value)
            for feature, value in importances.items()
        }
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("             THREATLENS X - MODEL TRAINING")
    print("=" * 70)

    email_results = train_email_model()

    malware_results = train_malware_model()

    final_report = {
        "email_model": email_results,
        "malware_model": malware_results
    }

    report_file = (
        REPORT_DIR /
        "model_evaluation.json"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_report,
            f,
            indent=4
        )

    print("\n" + "=" * 70)
    print("              TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"\nEvaluation report:\n{report_file}"
    )