from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import config

class SentimentAnalyzer:
    def _init_(self):
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        print("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.MODEL_NAME)
        self.model.to(self.device)
        print("Model loaded!")

    def is_loaded(self):
        return self.model is not None and self.tokenizer is not None

    def analyze(self, text: str):
        if not self.is_loaded():
            raise Exception("Model not loaded yet")

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=config.MAX_LENGTH)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence, label = torch.max(probs, dim=1)

        sentiment = "POSITIVE" if label.item() == 1 else "NEGATIVE"
        return {"text": text, "sentiment": sentiment, "confidence": float(confidence.item())}