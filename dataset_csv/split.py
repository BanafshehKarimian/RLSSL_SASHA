
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def get_patient_number(case_id):
    match = re.fullmatch(r"patient_(\d+)", str(case_id))
    if match is None:
        raise ValueError(f"Invalid case_id: {case_id}")
    return int(match.group(1))


def main():
    parser = argparse.ArgumentParser(
        description="Generate 5 patient-stratified CAMELYON17 splits."
    )
    parser.add_argument("--csv_path", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0 < args.val_frac < 1:
        raise ValueError("--val_frac must be between 0 and 1.")

    df = pd.read_csv(args.csv_path)

    required = {"case_id", "slide_id", "label", "center"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df["slide_id"].isna().any() or df["case_id"].isna().any():
        raise ValueError("Missing case_id or slide_id values.")

    if df["slide_id"].duplicated().any():
        raise ValueError("Duplicate slide_id values found.")

    if not df["label"].isin([0, 1]).all():
        raise ValueError("Labels must be binary: 0 or 1.")

    df = df.copy()
    df["patient_number"] = df["case_id"].map(get_patient_number)

    if not df["patient_number"].between(0, 199).all():
        raise ValueError("Patient IDs must be between 000 and 199.")

    # Official CAMELYON17 patient-number partition.
    dev_df = df[df["patient_number"] < 100].copy()
    test_df = df[df["patient_number"] >= 100].copy()

    if dev_df.empty or test_df.empty:
        raise ValueError(
            "Both development (000-099) and test (100-199) "
            "slides must be present in the input CSV."
        )

    # Patient-level labels: positive if any slide is positive.
    patients = (
        dev_df.groupby("case_id", sort=True)["label"]
        .max()
        .reset_index()
    )

    patient_ids = patients["case_id"].to_numpy()
    patient_labels = patients["label"].to_numpy()

    class_counts = patients["label"].value_counts().to_dict()

    print(f"Input: {args.csv_path}")
    print(f"Development: {len(dev_df)} slides, "
          f"{len(patients)} patients")
    print(f"Test: {len(test_df)} slides, "
          f"{test_df['case_id'].nunique()} patients")
    print(f"Development patient labels: {class_counts}")

    # StratifiedShuffleSplit requires at least two patients
    # per class and enough patients in each partition.
    if len(class_counts) != 2 or min(class_counts.values()) < 2:
        raise ValueError(
            "Insufficient patients per class for stratified splitting."
        )

    splitter = StratifiedShuffleSplit(
        n_splits=args.n_splits,
        test_size=args.val_frac,
        random_state=args.seed,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    test_names = test_df["slide_id"].astype(str).tolist()

    for split_idx, (train_idx, val_idx) in enumerate(
        splitter.split(patient_ids, patient_labels), start=1
    ):
        train_patients = set(patient_ids[train_idx])
        val_patients = set(patient_ids[val_idx])

        train_df = dev_df[
            dev_df["case_id"].isin(train_patients)
        ]
        val_df = dev_df[
            dev_df["case_id"].isin(val_patients)
        ]

        train_names = train_df["slide_id"].astype(str).tolist()
        val_names = val_df["slide_id"].astype(str).tolist()

        # Leakage checks.
        assert train_patients.isdisjoint(val_patients)
        assert set(train_names).isdisjoint(val_names)
        assert set(train_names).isdisjoint(test_names)
        assert set(val_names).isdisjoint(test_names)

        split = {
            "train_names": train_names,
            "val_names": val_names,
            "test_names": test_names,
        }

        output_path = args.out_dir / f"split_{split_idx}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(split, f, indent=2)

        print(
            f"Split {split_idx}: "
            f"train={len(train_names)} slides "
            f"({len(train_patients)} patients), "
            f"val={len(val_names)} slides "
            f"({len(val_patients)} patients), "
            f"test={len(test_names)} slides"
        )

    print(f"\nSaved {args.n_splits} splits to {args.out_dir}")


if __name__ == "__main__":
    main()
