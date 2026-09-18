from pathlib import Path
import pandas as pd
import json

# ============================================================
# THREATLENS X - DATASET INSPECTION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("             THREATLENS X - DATASET INSPECTION")
print("=" * 70)

print(f"\nProject directory : {PROJECT_ROOT}")
print(f"Data directory    : {DATA_DIR}")

if not DATA_DIR.exists():
    print("\nERROR: data folder does not exist.")
    print(f"Create it here: {DATA_DIR}")
    raise SystemExit(1)


# ------------------------------------------------------------
# Find supported files recursively
# ------------------------------------------------------------

supported_extensions = {
    ".csv",
    ".xlsx",
    ".xls",
    ".json"
}

files = [
    file for file in DATA_DIR.rglob("*")
    if file.is_file() and file.suffix.lower() in supported_extensions
]

if not files:
    print("\nNo CSV/Excel/JSON files found.")
    print("Check that you extracted your datasets inside:")
    print(DATA_DIR)
    raise SystemExit(1)


print(f"\nFound {len(files)} dataset file(s).\n")


# ------------------------------------------------------------
# Store overall report
# ------------------------------------------------------------

full_report = {}


# ------------------------------------------------------------
# Analyze each file
# ------------------------------------------------------------

for file_path in files:

    relative_path = file_path.relative_to(DATA_DIR)

    print("\n" + "=" * 70)
    print(f"FILE: {relative_path}")
    print("=" * 70)

    try:

        # -----------------------------
        # Load dataset
        # -----------------------------

        extension = file_path.suffix.lower()

        if extension == ".csv":
            df = pd.read_csv(file_path, low_memory=False)

        elif extension in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)

        elif extension == ".json":
            df = pd.read_json(file_path)

        else:
            continue

        # -----------------------------
        # Basic information
        # -----------------------------

        rows, columns = df.shape

        print(f"\nRows       : {rows:,}")
        print(f"Columns    : {columns}")

        print("\nColumns:")
        for column in df.columns:
            print(f"  - {column}")

        # -----------------------------
        # Data types
        # -----------------------------

        print("\nData types:")
        print(df.dtypes.to_string())

        # -----------------------------
        # Missing values
        # -----------------------------

        missing = df.isnull().sum()

        print("\nMissing values:")

        missing_found = False

        for column, count in missing.items():

            if count > 0:
                missing_found = True

                percentage = (count / rows) * 100

                print(
                    f"  {column}: "
                    f"{count:,} ({percentage:.2f}%)"
                )

        if not missing_found:
            print("  No missing values.")

        # -----------------------------
        # Duplicate rows
        # -----------------------------

        duplicates = df.duplicated().sum()

        print(f"\nDuplicate rows: {duplicates:,}")

        # -----------------------------
        # Possible label columns
        # -----------------------------

        possible_labels = []

        label_keywords = [
            "label",
            "spam",
            "class",
            "target",
            "legitimate",
            "malicious",
            "category",
            "type"
        ]

        for column in df.columns:

            column_lower = str(column).lower()

            if any(
                keyword in column_lower
                for keyword in label_keywords
            ):
                possible_labels.append(column)

        print("\nPossible label columns:")

        if possible_labels:

            for column in possible_labels:

                print(f"\n  {column}")

                unique_values = df[column].value_counts(
                    dropna=False
                )

                print(
                    unique_values.head(20).to_string()
                )

        else:
            print("  None detected.")

        # -----------------------------
        # Numeric columns
        # -----------------------------

        numeric_columns = df.select_dtypes(
            include="number"
        ).columns.tolist()

        print("\nNumeric columns:")

        if numeric_columns:
            print(
                "  " +
                ", ".join(map(str, numeric_columns))
            )
        else:
            print("  None")

        # -----------------------------
        # Text columns
        # -----------------------------

        text_columns = df.select_dtypes(
            include=["object", "string"]
        ).columns.tolist()

        print("\nText columns:")

        if text_columns:
            print(
                "  " +
                ", ".join(map(str, text_columns))
            )
        else:
            print("  None")

        # -----------------------------
        # First 3 records
        # -----------------------------

        print("\nFirst 3 records:")

        print(
            df.head(3).to_string(
                max_cols=20,
                max_colwidth=60
            )
        )

        # -----------------------------
        # Store report
        # -----------------------------

        dataset_report = {
            "file": str(relative_path),
            "rows": int(rows),
            "columns": int(columns),
            "column_names": [
                str(column)
                for column in df.columns
            ],
            "data_types": {
                str(column): str(dtype)
                for column, dtype in df.dtypes.items()
            },
            "missing_values": {
                str(column): int(count)
                for column, count in missing.items()
                if count > 0
            },
            "duplicate_rows": int(duplicates),
            "possible_label_columns": [
                str(column)
                for column in possible_labels
            ],
            "numeric_columns": [
                str(column)
                for column in numeric_columns
            ],
            "text_columns": [
                str(column)
                for column in text_columns
            ]
        }

        # Add label distributions

        label_distributions = {}

        for column in possible_labels:

            values = df[column].value_counts(
                dropna=False
            ).head(20)

            label_distributions[str(column)] = {
                str(index): int(value)
                for index, value in values.items()
            }

        dataset_report[
            "label_distributions"
        ] = label_distributions

        full_report[str(relative_path)] = dataset_report

    except Exception as error:

        print("\nERROR while reading file:")
        print(error)

        full_report[str(relative_path)] = {
            "file": str(relative_path),
            "error": str(error)
        }


# ------------------------------------------------------------
# Save JSON report
# ------------------------------------------------------------

report_file = REPORT_DIR / "dataset_inspection.json"

with open(
    report_file,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        full_report,
        file,
        indent=4,
        ensure_ascii=False
    )


# ------------------------------------------------------------
# Final message
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATASET INSPECTION COMPLETE")
print("=" * 70)

print(f"\nReport saved to:")
print(report_file)

print("\nNext step:")
print("Send me the terminal output from this script.")