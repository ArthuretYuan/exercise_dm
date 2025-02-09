# This script trains a Sentence-BERT Sentence Transformer model using the 'all-mpnet-base-v2' pre-trained model

import csv
import json
from sklearn.model_selection import train_test_split
import torch
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from datasets import Dataset


# load context code mapping
with open('data/code_context_map_v1.json', 'r') as f:
    context_code_mapping = json.load(f)
context_mode = 'text'  # 'code' or 'text'
output_dir = "model/sbert_st_text" if context_mode == 'text' else "model/sbert_st_code"


# Load the data
data, labels = [], []
with open('data/data.csv', newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(csvreader)  # Skip the header (id,main,target,context,score)
    for row in csvreader:
        if context_mode == 'code':
            data.append([row[1], row[2], row[3]])
        elif context_mode == 'text':
            data.append([row[1], row[2], context_code_mapping[row[3]]])
        labels.append(float(row[4]))

# Split the data into train, validation, and test sets (80% train, 10% validation, 10% test)
train_data, val_test_data, train_labels, val_test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=42)
val_data, test_data, val_labels, test_labels = train_test_split(
    val_test_data, val_test_labels, test_size=0.5, random_state=42)

# Train the model
# REF: https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss
# REF: https://sbert.net/docs/sentence_transformer/pretrained_models.html
# 'all-mpnet-base-v2' is a pre-trained sentence transformer based on 'microsoft/mpnet-base', the size is 420MB
model = SentenceTransformer("all-mpnet-base-v2")

train_sentence1 = [f'{texts[0]} [SEP] {texts[2]}' for texts in train_data]
train_sentence2 = [f'{texts[1]} [SEP] {texts[2]}' for texts in train_data]
train_dataset = Dataset.from_dict({
    "sentence1": train_sentence1,
    "sentence2": train_sentence2,
    "score": train_labels,
})
val_sentence1 = [f'{texts[0]} [SEP] {texts[2]}' for texts in val_data]
val_sentence2 = [f'{texts[1]} [SEP] {texts[2]}' for texts in val_data]
val_dataset = Dataset.from_dict({
    "sentence1": val_sentence1,
    "sentence2": val_sentence2,
    "score": val_labels,
})
loss = losses.CosineSimilarityLoss(model)

# Define the training arguments
# REF: https://www.sbert.net/docs/package_reference/sentence_transformer/training_args.html#sentence_transformers.training_args.SentenceTransformerTrainingArguments
args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    num_train_epochs=6,
    save_steps=10000,
    logging_steps=1000)

trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=args,
    loss=loss,
)
trainer.train()

# Save the model
torch.save(model, f'{output_dir}/sbert_st_model.pt')