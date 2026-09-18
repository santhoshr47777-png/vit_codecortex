from pathlib import Path
from io import BytesIO
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_csv(path):
    """
    Automatically handles normal CSV and pipe-separated
    malware CSV.
    """

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()

    if "|" in first_line and "," not in first_line:
        return pd.read_csv(
            path,
            sep="|",
            low_memory=False
        )

    return pd.read_csv(
        path,
        low_memory=False
    )


def find_label_column(df):

    possible = [
        "label",
        "spam",
        "legitimate",
        "class",
        "target"
    ]

    for column in possible:
        if column in df.columns:
            return column

    return None


def find_text_columns(df):

    keywords = [
        "text",
        "subject",
        "body",
        "message",
        "content",
        "combined"
    ]

    result = []

    for column in df.columns:

        name = str(column).lower()

        if any(keyword in name for keyword in keywords):
            result.append(column)

    return result


print("=" * 75)
print("             THREATLENS X - LABEL ANALYSIS")
print("=" * 75)

files = list(DATA_DIR.rglob("*.csv"))

if not files:
    print("No CSV files found.")
    raise SystemExit(1)


for path in files:

    relative = path.relative_to(DATA_DIR)

    print("\n" + "=" * 75)
    print(f"DATASET: {relative}")
    print("=" * 75)

    try:

        df = load_csv(path)

        print(f"\nRows: {len(df):,}")
        print(f"Columns: {len(df.columns)}")

        label_column = find_label_column(df)

        print(f"\nLabel column: {label_column}")

        if label_column is not None:

            print("\nLABEL VALUES:")

            counts = df[label_column].value_counts(
                dropna=False
            )

            for value, count in counts.items():

                percentage = count / len(df) * 100

                print(
                    f"  {repr(value):20} "
                    f"{count:8,} "
                    f"({percentage:6.2f}%)"
                )

        text_columns = find_text_columns(df)

        print("\nTEXT COLUMNS:")

        if text_columns:

            for column in text_columns:
                print(f"  - {column}")

        else:
            print("  None")

        # ----------------------------------------------------
        # Show examples grouped by label
        # ----------------------------------------------------

        if label_column and text_columns:

            text_column = text_columns[0]

            print("\nEXAMPLES BY LABEL:")

            unique_labels = (
                df[label_column]
                .dropna()
                .unique()
            )

            for label in unique_labels:

                subset = df[
                    df[label_column] == label
                ]

                if len(subset) == 0:
                    continue

                print(
                    f"\n--- LABEL {repr(label)} ---"
                )

                sample = subset[
                    text_column
                ].iloc[0]

                if pd.isna(sample):
                    print("[EMPTY]")

                else:

                    sample = str(sample)

                    print(
                        sample[:700]
                        .replace("\n", " ")
                    )

        # ----------------------------------------------------
        # Check empty text
        # ----------------------------------------------------

        if text_columns:

            print("\nEMPTY TEXT ANALYSIS:")

            for column in text_columns:

                empty = (
                    df[column]
                    .isna()
                    .sum()
                )

                if df[column].dtype == "object":

                    empty += (
                        df[column]
                        .astype(str)
                        .str.strip()
                        .eq("")
                        .sum()
                    )

                percentage = (
                    empty / len(df) * 100
                )

                print(
                    f"  {column}: "
                    f"{empty:,} "
                    f"({percentage:.2f}%)"
                )

    except Exception as e:

        print("\nERROR:")
        print(e)


print("\n" + "=" * 75)
print("LABEL ANALYSIS COMPLETE")
print("=" * 75)