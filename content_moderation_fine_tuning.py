import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import evaluate
from peft import get_peft_model, LoraConfig, TaskType

# Check if CUDA is available and use it
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Load the dataset
data_dir = '/Users/taehoonkang/Downloads/jigsaw-toxic-comment-classification'  # Ensure this path is correct
dataset = load_dataset('csv', data_files={'train': f'{data_dir}/train.csv', 'test': f'{data_dir}/test.csv'})

# Prepare the dataset (keeping only relevant columns and renaming for consistency)
def prepare_dataset(examples):
    examples['label'] = examples['toxic']
    examples['text'] = examples['comment_text']
    return examples

dataset = dataset.map(prepare_dataset, remove_columns=['severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])

# Split the dataset into train and test sets
dataset = dataset['train'].train_test_split(test_size=0.2)

# Load a pre-trained tokenizer and model (GPT-2)
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)  # Move the model to the appropriate device

# Apply LoRA for efficient fine-tuning
config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, target_modules=["c_attn", "c_proj"], lora_dropout=0.1)
model = get_peft_model(model, config)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
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

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()

# Save the model
model.save_pretrained("./content_moderation_model")
tokenizer.save_pretrained("./content_moderation_model")

# Function to perform inference
def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)  # Move inputs to device
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return "Inappropriate" if predictions.item() == 1 else "Appropriate"

# Prompt the user for input text
user_input = input("Enter a text to classify: ")
classification = classify_text(user_input)
print(f"Classification: {classification}")
