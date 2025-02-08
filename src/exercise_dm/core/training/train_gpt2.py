import csv
import json
import torch

from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split


# tokenizer
def encode_sentences(data, tokenizer):
    inputs = [f"{texts[0]} <|sep|> {texts[1]} <|sep|> {texts[2]}" for texts in data]
    return tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")

# Dataset class
class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)



# load context code mapping
with open('data/code_context_map_v1.json', 'r') as f:
    context_code_mapping = json.load(f)

# Load the data
data, labels = [], []

with open('data/data.csv', newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(csvreader)  # Skip the header (id,main,target,context,score)

    for row in csvreader:
        data.append([row[1], row[2], context_code_mapping[row[3]].lower()])
        labels.append(float(row[4]))

# Split the data into train, validation, and test sets (80% train, 10% validation, 10% test)
train_data, val_test_data, train_labels, val_test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=42
)
val_data, test_data, val_labels, test_labels = train_test_split(
    val_test_data, val_test_labels, test_size=0.5, random_state=42
)



# Fix padding issue
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Use <|endoftext|> as padding
train_encodings = encode_sentences(train_data, tokenizer)
train_labels = torch.tensor(train_labels).unsqueeze(1)  # Convert to tensor
val_encodings = encode_sentences(val_data, tokenizer)
val_labels = torch.tensor(val_labels).unsqueeze(1)  # Convert to tensor


train_dataset = NewsDataset(train_encodings, train_labels)
val_dataset = NewsDataset(val_encodings, val_labels)


# model selection, REF: https://huggingface.co/transformers/v2.2.0/pretrained_models.html
model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=1)  # Regression task


training_args = TrainingArguments(
    output_dir="model/gpt2/gpt2-similarity",
    per_device_train_batch_size=16,
    learning_rate=5e-5,
    num_train_epochs=6,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="model/gpt2/logs",
    logging_steps=100,
)

torch.set_grad_enabled(True)  # Context-manager
model.config.pad_token_id = model.config.eos_token_id
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

torch.save(model, 'model/gpt2/gpt2_model.pt')