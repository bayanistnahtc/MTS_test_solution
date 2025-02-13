import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline
)
from peft import PeftModel, PeftConfig


class QueryClassifierPeft:
    def __init__(self, model_path):

        config = PeftConfig.from_pretrained(model_path)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model_name_or_path,
            num_labels=2
        )

        # Load LoRA adapters
        model = PeftModel.from_pretrained(base_model, model_path)

        # Merge adapters with base model for inference
        model = model.merge_and_unload()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name_or_path
        )

        self.classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )

    def predict(self, query: str, threshold: float = 0.5):
        result = self.classifier(
            query,
            truncation=True,
            max_length=512
        )
        return {
            "decision": "retrieval required" if result[0]['label'] == "LABEL_1"
                        else "no retrieval required",
            "confidence": result[0]['score']
        }


if __name__ == "__main__":
    query_classifier_model = QueryClassifierPeft("/query_classifier")
examples = [
    "What is the capital of France?", # no retrival required (general knowledge)
    "Who is the current president of FIFA?", # retrival required (the data must be actual)
    "write summary of last news", # retrival required (the news are not provided)
    "What is the boiling point of water?", # no retrieval required (scientific fact)
    "What are the upcoming concerts in my city?", # retrieval required (local event information)
    "What is the chemical formula for water?" # no retrieval required (chemical fact)
]


query_classifier_model.predict(examples[0])
