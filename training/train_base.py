from dataclasses import dataclass
from typing import List, Union

import datasets
import evaluate
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)


@dataclass
class TrainConfig:
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    max_length: int = 512
    test_size: float = 0.1
    batch_size: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    epochs: int = 10
    seed: int = 42


class WeightedTrainer(Trainer):
    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(model.device)
        )
        loss = loss_fct(outputs.logits.view(-1, 2), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def preprocess_function(
    examples,
    tokenizer,
    padding: str = "max_length",
    truncation: bool = True,
    max_length: int = 512,
    return_tensors: str = "pt"
):
    texts = []
    for instr, ctx in zip(examples['instruction'], examples['context']):
        if ctx and len(ctx) > 0:
            combined = f"{instr}" #\n\n{ctx}"
        else:
            combined = f"{instr}"
        texts.append(combined)
    return tokenizer(
        texts,
        padding=padding,
        truncation=truncation,
        max_length=max_length,
        return_tensors=return_tensors
    )


def compute_metrics(eval_pred):
    metrics = {
        "f1": evaluate.load("f1"),
        "precision": evaluate.load("precision"),
        "recall": evaluate.load("recall"),
    }
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    results = {
        "f1_macro": metrics["f1"].compute(
            predictions=preds, references=labels, average="macro")["f1"],
        "precision": metrics["precision"].compute(
            predictions=preds, references=labels, average="binary")["precision"],
        "recall": metrics["recall"].compute(
            predictions=preds, references=labels, average="binary")["recall"],
    }
    return results


def train_model(
        config: TrainConfig,
        dataset_paths: Union[str, List[str]],
        output_dir: str
    ):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if isinstance(dataset_paths, list):
        dataset_list = []
        for path in dataset_paths:
            ds = datasets.load_from_disk(path)
            dataset_list.append(ds)
        dataset = datasets.concatenate_datasets(dataset_list)
    else:
        dataset = datasets.load_from_disk(dataset_paths)

    dataset = dataset.rename_column("retrieval_label", "labels")
    dataset = dataset.class_encode_column("labels")

    train_test = dataset.train_test_split(
        test_size=config.test_size,
        stratify_by_column="labels",
        seed=config.seed
    )

    # Initialize model
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2
    )

    # Class weights
    class_counts = np.bincount(train_test["train"]["labels"])
    class_weights = torch.tensor([
        1.0/class_counts[0],
        1.0/class_counts[1]
    ], dtype=torch.float32)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size*2,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        save_total_limit=2,
        report_to="none",
    )

    # Trainer
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_test["train"].map(
            preprocess_function,
            batched=True,
            fn_kwargs={
                "tokenizer": tokenizer,
                "max_length": config.max_length
                }),
        eval_dataset=train_test["test"].map(
            preprocess_function,
            batched=True,
            fn_kwargs={
                "tokenizer": tokenizer,
                "max_length": config.max_length
                }),
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model(f"{output_dir}/best_model")

    return trainer
