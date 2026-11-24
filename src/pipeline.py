import hashlib
import json
from pathlib import Path
import pandas as pd


def sha256_of_file(path):
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def build_manifest(raw_dir, out_path):
    csv_files = sorted(Path(raw_dir).glob("*.csv"))
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    manifest = {}
    for file in csv_files:
        manifest[file.name] = {
            "sha256": sha256_of_file(file),
            "bytes": file.stat().st_size,
        }

    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest

LABEL_MAP = {"0": "safe", "1": "attack", "True": "attack", "False": "safe"}

def verify_manifest(raw_dir, manifest_path):
    with open(manifest_path) as f:
        saved = json.load(f)
    for name, info in saved.items():
        current = sha256_of_file(Path(raw_dir) / name)
        if current != info["sha256"]:
            raise ValueError(f"{name} has changed since the manifest was made")
    return True


def load_source(raw_dir, file_name, source_dataset, source_split,
                text_col, label_col, detail_col=None):
    
    df = pd.read_csv(Path(raw_dir) / file_name, keep_default_na=False)
    out = pd.DataFrame()
    out["text"] = df[text_col]
    out["source_dataset"] = source_dataset
    out["source_split"] = source_split
    out["source_row_id"] = range(len(df))
    out["original_label"] = df[label_col].astype(str)
    out["source_detail"] = df[detail_col] if detail_col else ""
    return out

def build_stage1(raw_dir, out_path):
    parts = [
        load_source(raw_dir, "deepset_train.csv", "deepset", "train", "text", "label"),
        load_source(raw_dir, "deepset_test.csv", "deepset", "test", "text", "label"),
        load_source(raw_dir, "safeguard_train.csv", "safeguard", "train", "text", "label"),
        load_source(raw_dir, "safeguard_test.csv", "safeguard", "test", "text", "label"),
        load_source(raw_dir, "itw_jailbreak.csv", "itw_jailbreak", "all", "prompt", "jailbreak", "source"),
        load_source(raw_dir, "itw_regular.csv", "itw_regular", "all", "prompt", "jailbreak", "source"),
    ]
    df = pd.concat(parts, ignore_index=True)

    df["coarse_label"] = df["original_label"].map(LABEL_MAP)
    if df["coarse_label"].isna().any():
        raise ValueError("Found an original label that is not in LABEL_MAP")

    ids = (df["source_dataset"] + "-" + df["source_split"] + "-"
           + df["source_row_id"].astype(str).str.zfill(6))
    df.insert(0, "id", ids)
    if df["id"].duplicated().any():
        raise ValueError("Duplicate ids found")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df

def build_stage2(in_path, out_path):
    df = pd.read_csv(in_path, dtype=str, keep_default_na=False)

    length = df["text"].str.len()
    norm = df["text"].str.lower().str.split().str.join(" ")

    df["is_short"] = length < 5
    df["is_long"] = length > 5000
    df["dup_group_id"] = "g" + norm.groupby(norm).ngroup().astype(str).str.zfill(6)
    df["dup_count"] = norm.groupby(norm).transform("size")
    df["label_conflict"] = df.groupby(norm)["coarse_label"].transform("nunique") > 1

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    return df