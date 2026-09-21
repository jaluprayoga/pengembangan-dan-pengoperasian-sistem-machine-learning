import os
from typing import Tuple
import pandas as pd

DATA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)

COLUMN_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]


def download_diabetes_dataset(output_path: str) -> pd.DataFrame:
    """Download and format the Pima Indians Diabetes dataset.

    Args:
        output_path (str): Destination file path for the CSV output.

    Returns:
        pd.DataFrame: Formatted diabetes dataset DataFrame.
    """
    print(f"Downloading diabetes dataset from: {DATA_URL}")
    df = pd.read_csv(DATA_URL, header=None, names=COLUMN_NAMES)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Ensure correct data types
    for col in COLUMN_NAMES[:-1]:
        df[col] = df[col].astype(float)
    df["Outcome"] = df["Outcome"].astype(int)

    # Save to CSV without index
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully saved to: {output_path}")
    print(f"Dataset shape: {df.shape}")
    print("Label distribution (Outcome):\n", df["Outcome"].value_counts())

    return df


def main() -> Tuple[str, int]:
    """Execute main dataset download process.

    Returns:
        Tuple[str, int]: Saved dataset path and total number of records.
    """
    target_path = os.path.join("data", "diabetes.csv")
    df = download_diabetes_dataset(target_path)
    return target_path, len(df)


if __name__ == "__main__":
    main()
