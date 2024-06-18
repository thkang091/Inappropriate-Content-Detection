import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

data_dir = '/Users/taehoonkang/Downloads/Inappropriate-Content-Detection/jigsaw-toxic-comment-classification'  # Ensure this path is correct

dataset = load_dataset('jigsaw_toxicity_pred', data_dir=data_dir, trust_remote_code=True)

def prepare_dataset(examples):
    examples['label'] = examples['toxic']
    examples['tweet'] = examples['comment_text']
    return examples

dataset = dataset.map(prepare_dataset, remove_columns=['toxic', 'comment_text'])

dataset = dataset['train'].train_test_split(test_size=0.2)

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)  

def tokenize_function(examples):
    return tokenizer(examples['tweet'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,  # Adjust the number of epochs if necessary
    weight_decay=0.01,
)

# Define the metrics
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)  # Convert logits to tensor before applying argmax
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

trainer.train()

model.save_pretrained("./content_moderation_model")
tokenizer.save_pretrained("./content_moderation_model")

def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)  # Move inputs to device
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return "Inappropriate" if predictions.item() == 1 else "Appropriate"

user_input = input("Enter a text to classify: ")
classification = classify_text(user_input)
print(f"Classification: {classification}")