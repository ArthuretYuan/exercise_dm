# This script trains a GPT2 model using the 'gpt2' pre-trained model

import csv
import json
import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from exercise_dm.settings import PRE_TRAINED_MODEL_GPT2

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

# tokenizer
def encode_sentences(data, tokenizer):
    inputs = [f"{texts[0]} <|sep|> {texts[1]} <|sep|> {texts[2]}" for texts in data]
    return tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")

# load data
def load_data(data_path, context_codes_path, context_mode):

    # load context code mapping
    with open(context_codes_path, 'r') as f:
        context_code_mapping = json.load(f)

    # load data and labels
    data, labels = [], []
    with open(data_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(csvreader)  # Skip the header (id,main,target,context,score)
        for row in csvreader:
            if context_mode == 'code':
                data.append([row[1], row[2], row[3]])
            elif context_mode == 'text':
                data.append([row[1], row[2], context_code_mapping[row[3]].lower()])
            labels.append(float(row[4]))
    return data, labels

# train gpt2 model
def train_gpt2(data_path, context_codes_path, context_mode, output_path):
    
    data, labels = load_data(data_path, context_codes_path, context_mode)

    # Split the data into train, validation, and test sets (80% train, 10% validation, 10% test)
    train_data, val_test_data, train_labels, val_test_labels = train_test_split(
        data, labels, test_size=0.2, random_state=42)
    val_data, _, val_labels, _ = train_test_split(
        val_test_data, val_test_labels, test_size=0.5, random_state=42)

    tokenizer = GPT2Tokenizer.from_pretrained(PRE_TRAINED_MODEL_GPT2)
    tokenizer.pad_token = tokenizer.eos_token  # Use <|endoftext|> as padding
    train_encodings = encode_sentences(train_data, tokenizer)
    train_labels = torch.tensor(train_labels).unsqueeze(1)  # Convert to tensor
    val_encodings = encode_sentences(val_data, tokenizer)
    val_labels = torch.tensor(val_labels).unsqueeze(1)  # Convert to tensor

    train_dataset = NewsDataset(train_encodings, train_labels)
    val_dataset = NewsDataset(val_encodings, val_labels)

    # select model, REF: https://huggingface.co/transformers/v2.2.0/pretrained_models.html
    model = GPT2ForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_GPT2, num_labels=1)  # Regression task

    training_args = TrainingArguments(
        output_dir=f"{output_path}/gpt2-similarity",
        per_device_train_batch_size=16,
        learning_rate=5e-5,
        num_train_epochs=15,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_path}/logs",
        logging_steps=100,
    )

    torch.set_grad_enabled(True)
    model.config.pad_token_id = model.config.eos_token_id
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # start training
    trainer.train()

    # save final model
    torch.save(model, f"{output_path}/gpt2_model.pt")


if __name__ == '__main__':
    DATA_PATH = 'data/data.csv'
    CONTEXT_CODES_PATH = 'data/code_context_map_v1.json'
    CONTEXT_MODE = 'text'  # 'code' or 'text'
    OUTPUT_PATH = f"model/sbert_ce_{CONTEXT_MODE}"
    train_gpt2(data_path=DATA_PATH, 
               context_codes_path=CONTEXT_CODES_PATH,
               context_mode=CONTEXT_MODE,
               output_path=OUTPUT_PATH)