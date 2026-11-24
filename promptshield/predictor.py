"""Load a model once and turn a prompt into a label and an attack score."""
from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from promptshield.registry import MAX_CHARS, MAX_TOKENS, SUPPORTED_KINDS, load_thresholds, source_for


class PredictionError(Exception):
    """An error whose message is safe to show to the user."""


@dataclass(frozen=True)
class Prediction:
    label: str          # "Safe" or "Attack"
    score: float        # attack score between 0 and 1
    threshold: float    # the validation cut-off used for this model
    model_label: str
    truncated: bool     # True if only the beginning of the prompt was analysed
    n_tokens: int       # prompt length in tokens before cutting


class Predictor:
    def __init__(self, spec):
        if spec.kind not in SUPPORTED_KINDS:
            raise PredictionError(f"{spec.label} is not enabled in this version of the app.")
        thresholds = load_thresholds()
        if spec.key not in thresholds:
            raise PredictionError(f"No validation threshold is saved for {spec.label}.")
        source = source_for(spec)
        if source is None:
            raise PredictionError(f"The files for {spec.label} were not found.")

        self.spec = spec
        self.threshold = thresholds[spec.key]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = AutoModelForSequenceClassification.from_pretrained(source).to(self.device)
        self.model.eval()

    def predict(self, text):
        text = (text or "").strip()
        if not text:
            raise PredictionError("Please enter a prompt to analyze.")
        try:
            chars_cut = len(text) > MAX_CHARS
            text = text[:MAX_CHARS]
            n_tokens = len(self.tokenizer(text)["input_ids"])

            enc = self.tokenizer(text, truncation=True, max_length=MAX_TOKENS, return_tensors="pt")
            with torch.no_grad():
                logits = self.model(
                    input_ids=enc["input_ids"].to(self.device),
                    attention_mask=enc["attention_mask"].to(self.device),
                ).logits
            score = float(torch.softmax(logits, dim=-1)[0, 1])
        except Exception as error:
            raise PredictionError("The model could not score this prompt. Please try again.") from error

        return Prediction(
            label="Attack" if score > self.threshold else "Safe",
            score=score,
            threshold=self.threshold,
            model_label=self.spec.label,
            truncated=chars_cut or n_tokens > MAX_TOKENS,
            n_tokens=n_tokens,
        )
        