
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate
import numpy as np

"""
Loads data
It downloads the IMDb dataset with train/test splits.

Loads tokenizer
It uses bert-base-uncased tokenizer and defines a tokenize function that:

truncates long reviews
pads to fixed length
limits sequence length to 256 tokens
Prepares tensors
It tokenizes all samples, renames label to labels (Trainer expects this), and formats tensors as PyTorch inputs:
input_ids
attention_mask
labels
Uses smaller subsets
For faster runs, it keeps:
5,000 training samples
1,000 test samples
Loads model
It initializes bert-base-uncased for sequence classification with 2 output labels (negative/positive).

Defines metrics
During evaluation, it computes:

accuracy
binary F1 score
Predictions come from argmax over model logits.
Sets training configuration
Key settings:
output/logging directories
evaluate and save each epoch
batch size 8
2 epochs
learning rate 2e-5
weight decay 0.01
keep best model at end
disable external reporting
Trains and evaluates
Trainer handles training loop, then runs evaluate() and prints metrics.

Saves artifacts
It saves both fine-tuned model and tokenizer to ./bert-sentiment-model.

"""

# 1. Load dataset
dataset = load_dataset("imdb")

# 2. Load tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

tokenized = dataset.map(tokenize, batched=True)
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Optional: smaller subsets for faster testing
train_dataset = tokenized["train"].shuffle(seed=42).select(range(5000))
test_dataset = tokenized["test"].shuffle(seed=42).select(range(1000))

# 3. Load model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 4. Metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="binary")["f1"],
    }

# 5. Training config
training_args = TrainingArguments(
    output_dir="./bert-sentiment",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    report_to="none",
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# 7. Train
trainer.train()

# 8. Evaluate
results = trainer.evaluate()
print(results)

# 9. Save model
trainer.save_model("./bert-sentiment-model")
tokenizer.save_pretrained("./bert-sentiment-model")
