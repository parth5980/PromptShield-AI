"""Check that the app's predictor reproduces the saved VALIDATION probabilities.
Only validation rows are scored. The test split is not used."""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
csv.field_size_limit(10_000_000)

from promptshield.predictor import Predictor, PredictionError
from promptshield.registry import get_spec

N = 200

val_rows = []
with open(ROOT / "data" / "processed" / "dataset_v2.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["split"] != "val":
            continue
        val_rows.append((row["id"], row["text"]))
        if len(val_rows) == N:
            break

saved = {}
with open(ROOT / "results" / "tables" / "distilbert_v2_val_probs.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        saved[row["id"]] = float(row["prob_attack"])

predictor = Predictor(get_spec("DistilBERT"))
print(f"Model loaded: {predictor.spec.label} | threshold {predictor.threshold} | device: {predictor.device}")

diffs, label_mismatches = [], 0
for row_id, text in val_rows:
    result = predictor.predict(text)
    diffs.append(abs(result.score - saved[row_id]))
    saved_label = "Attack" if saved[row_id] > predictor.threshold else "Safe"
    label_mismatches += int(result.label != saved_label)

mean_diff, max_diff = sum(diffs) / len(diffs), max(diffs)
print(f"Checked {len(diffs)} validation prompts: mean abs score difference {mean_diff:.6f} | max {max_diff:.6f}")
print(f"Prompts whose label differs from the saved scores: {label_mismatches}")

for name, bad in [("Empty input", ""), ("Whitespace-only input", "   \n  ")]:
    try:
        predictor.predict(bad)
        print(name, "was NOT rejected")
        sys.exit(1)
    except PredictionError as error:
        print(f"{name} rejected: {error}")

long_result = predictor.predict("hello " * 2000)
print("Very long prompt | tokens before cutting:", long_result.n_tokens, "| truncated:", long_result.truncated)

ok = mean_diff < 0.002 and max_diff < 0.05 and label_mismatches <= 1 and long_result.truncated
print("All checks passed." if ok else "CHECK FAILED: do not continue until this is understood.")
sys.exit(0 if ok else 1)