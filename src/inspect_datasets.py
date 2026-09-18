from pathlib import Path
import pandas as pd
import json

# ============================================================
# THREATLENS X
# DATASET INSPECTION & VALIDATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(exist_ok=True)

print("=" * 75)
print("              THREATLENS X - DATASET INSPECTION")
print("=" * 75)

print(f"\nProject : {PROJECT_ROOT}")
print(f"Data    : {DATA_DIR}")

if not DATA_DIR.exists():
    print("\nERROR: data directory not found.")
    raise SystemExit(1)

# ------------------------------------------------------------
# Find CSV files
# ------------------------------------------------------------

files = list(DATA_DIR.rglob("*.csv"))

if not files:
    print("\nERROR: No CSV files found.")
    print("Put your extracted datasets inside the data folder.")
    raise SystemExit(1)

print(f"\nFound {len(files)} CSV files.")

report = {}

# ------------------------------------------------------------
# Load CSV intelligently
# ------------------------------------------------------------

def load_csv(file_path):

    # Read a small sample to detect separator
    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:
        sample = f.read(5000)

    # Malware dataset uses |
    if "|" in sample and "," not in sample.split("\n")[0]:
        return pd.read_csv(
            file_path,
            sep="|",
            low_memory=False
        )

    # Normal CSV
    return pd.read_csv(
        file_path,
        low_memory=False
    )


# ------------------------------------------------------------
# Inspect each dataset
# ------------------------------------------------------------

for file_path in files:

    relative = file_path.relative_to(DATA_DIR)

    print("\n" + "=" * 75)
    print(f"FILE: {relative}")
    print("=" * 75)

    try:

        df = load_csv(file_path)

        rows, cols = df.shape

        print(f"\nRows    : {rows:,}")
        print(f"Columns : {cols}")

        # ----------------------------------------------------
        # Columns
        # ----------------------------------------------------

        print("\nCOLUMNS")

        for column in df.columns:
            print(f"  {column}")

        # ----------------------------------------------------
        # Data types
        # ----------------------------------------------------

        print("\nDATA TYPES")

        print(df.dtypes.to_string())

        # ----------------------------------------------------
        # Missing values
        # ----------------------------------------------------

        print("\nMISSING VALUES")

        missing = df.isna().sum()

        missing_found = False

        for column, count in missing.items():

            if count > 0:

                missing_found = True

                percentage = (count / rows) * 100

                print(
                    f"  {column}: "
                    f"{count:,} "
                    f"({percentage:.2f}%)"
                )

        if not missing_found:
            print("  None")

        # ----------------------------------------------------
        # Duplicate rows
        # ----------------------------------------------------

        duplicates = df.duplicated().sum()

        print(f"\nDUPLICATE ROWS: {duplicates:,}")

        # ----------------------------------------------------
        # Possible label columns
        # ----------------------------------------------------

        label_keywords = [
            "label",
            "spam",
            "legitimate",
            "target",
            "class",
            "category",
            "malicious"
        ]

        label_columns = []

        for column in df.columns:

            column_name = str(column).lower()

            if any(
                keyword in column_name
                for keyword in label_keywords
            ):
                label_columns.append(column)

        print("\nPOSSIBLE LABEL COLUMNS")

        if label_columns:

            for column in label_columns:

                print(f"\n  {column}")

                counts = df[column].value_counts(
                    dropna=False
                )

                print(
                    counts.head(20).to_string()
                )

        else:
            print("  None detected")

        # ----------------------------------------------------
        # Numeric columns
        # ----------------------------------------------------

        numeric_columns = df.select_dtypes(
            include="number"
        ).columns.tolist()

        print("\nNUMERIC COLUMNS")

        print(
            f"  {len(numeric_columns)} numeric columns"
        )

        if numeric_columns:
            print(
                "  " +
                ", ".join(
                    map(str, numeric_columns)
                )
            )

        # ----------------------------------------------------
        # Text columns
        # ----------------------------------------------------

        text_columns = df.select_dtypes(
            include=["object", "string"]
        ).columns.tolist()

        print("\nTEXT COLUMNS")

        print(
            f"  {len(text_columns)} text columns"
        )

        if text_columns:
            print(
                "  " +
                ", ".join(
                    map(str, text_columns)
                )
            )

        # ----------------------------------------------------
        # Sample data
        # ----------------------------------------------------

        print("\nFIRST 3 ROWS")

        print(
            df.head(3).to_string(
                max_columns=12,
                max_colwidth=50
            )
        )

        # ----------------------------------------------------
        # Numerical statistics
        # ----------------------------------------------------

        if numeric_columns:

            print("\nNUMERICAL SUMMARY")

            print(
                df[numeric_columns]
                .describe()
                .transpose()
                .head(20)
                .to_string()
            )

        # ----------------------------------------------------
        # Save information
        # ----------------------------------------------------

        label_distribution = {}

        for column in label_columns:

            counts = df[column].value_counts(
                dropna=False
            )

            label_distribution[str(column)] = {
                str(index): int(value)
                for index, value in counts.items()
            }

        report[str(relative)] = {

            "rows": int(rows),

            "columns": int(cols),

            "column_names": [
                str(c)
                for c in df.columns
            ],

            "data_types": {
                str(c): str(dtype)
                for c, dtype in df.dtypes.items()
            },

            "missing_values": {
                str(c): int(v)
                for c, v in missing.items()
                if v > 0
            },

            "duplicate_rows": int(duplicates),

            "possible_label_columns": [
                str(c)
                for c in label_columns
            ],

            "label_distribution": label_distribution,

            "numeric_columns": [
                str(c)
                for c in numeric_columns
            ],

            "text_columns": [
                str(c)
                for c in text_columns
            ]
        }

    except Exception as error:

        print("\nERROR")
        print(error)

        report[str(relative)] = {
            "error": str(error)
        }


# ------------------------------------------------------------
# Save report
# ------------------------------------------------------------

output_file = REPORT_DIR / "dataset_inspection.json"

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=4,
        ensure_ascii=False
    )


# ------------------------------------------------------------
# Complete
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("             INSPECTION COMPLETE")
print("=" * 75)

print("\nReport:")
print(output_file)

print("\nNext:")
print("Send the terminal output to me.")