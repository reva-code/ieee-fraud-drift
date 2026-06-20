import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from config import DATA_DIR, BATCH_SIZE, REFERENCE_BATCHES, TOP_FEATURES


def load_and_merge() -> pd.DataFrame:
    # Load only the columns we actually use to avoid OOM on the 400-column full dataset
    tx_cols = ["TransactionID", "TransactionDT", "isFraud"] + [
        f for f in TOP_FEATURES if not f.startswith("id_")
    ]
    tx = pd.read_csv(DATA_DIR / "train_transaction.csv", usecols=lambda c: c in tx_cols)
    id_ = pd.read_csv(DATA_DIR / "train_identity.csv")
    df = tx.merge(id_, on="TransactionID", how="left")
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include="object").columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
    return df


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    return df


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    available = [f for f in TOP_FEATURES if f in df.columns]
    X = df[available].copy()
    y = df["isFraud"].copy()
    return X, y


def create_batches(X: pd.DataFrame, y: pd.Series) -> list[dict]:
    """Split stream into fixed-size batches preserving temporal order."""
    batches = []
    n = len(X)
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batches.append({
            "X": X.iloc[start:end].reset_index(drop=True),
            "y": y.iloc[start:end].reset_index(drop=True),
            "batch_id": len(batches),
        })
    return batches


def get_reference_window(batches: list[dict]) -> tuple[pd.DataFrame, pd.Series]:
    ref_X = pd.concat([b["X"] for b in batches[:REFERENCE_BATCHES]], ignore_index=True)
    ref_y = pd.concat([b["y"] for b in batches[:REFERENCE_BATCHES]], ignore_index=True)
    return ref_X, ref_y


def prepare_data() -> tuple[list[dict], pd.DataFrame, pd.Series]:
    print("Loading data...")
    df = load_and_merge()
    print(f"  Rows after merge: {len(df):,}")

    df = encode_categoricals(df)
    df = fill_missing(df)

    X, y = select_features(df)
    print(f"  Features selected: {X.shape[1]}")

    batches = create_batches(X, y)
    print(f"  Total batches: {len(batches)}  |  Batch size: {BATCH_SIZE}")

    ref_X, ref_y = get_reference_window(batches)
    print(f"  Reference window: {len(ref_X):,} samples ({REFERENCE_BATCHES} batches)")
    return batches, ref_X, ref_y
