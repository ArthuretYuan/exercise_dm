# This script trains a Sentence-BERT Sentence Transformer model using the 'all-mpnet-base-v2' pre-trained model

import csv
import json
import torch
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from datasets import Dataset
from settings import PRE_TRAINED_MODEL_SBERT_ST

# Load the data
def load_data(data_path, context_codes_path, context_mode):

    # load context code mapping
    with open(context_codes_path, 'r') as f:
        context_code_mapping = json.load(f)
    
    #load data and labels
    data, labels = [], []
    with open(data_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(csvreader)  # Skip the header (id,main,target,context,score)
        for row in csvreader:
            if context_mode == 'code':
                data.append([row[1], row[2], row[3]])
            elif context_mode == 'text':
                data.append([row[1], row[2], context_code_mapping[row[3]]])
            labels.append(float(row[4]))
    return data, labels


def train_sbert_st(data_path, context_codes_path, context_mode, output_path):


    # Split the data into train, validation, and test sets (80% train, 10% validation, 10% test)
    data, labels = load_data(data_path, context_codes_path, context_mode)
    train_data, val_test_data, train_labels, val_test_labels = train_test_split(
        data, labels, test_size=0.2, random_state=42)
    val_data, _, val_labels, _ = train_test_split(
        val_test_data, val_test_labels, test_size=0.5, random_state=42)

    # Re-format the data to the compatible format for Sentence-BERT Sentence Transformer
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

    # Model selection
    # REF: https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss
    # REF: https://sbert.net/docs/sentence_transformer/pretrained_models.html
    # 'all-mpnet-base-v2' is a pre-trained sentence transformer based on 'microsoft/mpnet-base', the size is 420MB
    model = SentenceTransformer(PRE_TRAINED_MODEL_SBERT_ST)
    loss = losses.CosineSimilarityLoss(model)

    # Define the training arguments
    # REF: https://www.sbert.net/docs/package_reference/sentence_transformer/training_args.html#sentence_transformers.training_args.SentenceTransformerTrainingArguments
    args = SentenceTransformerTrainingArguments(
        output_dir=output_path,
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
    torch.save(model, f'{output_path}/sbert_st_model.pt')


if __name__ == '__main__':
    DATA_PATH = 'data/data.csv'
    CONTEXT_CODES_PATH = 'data/code_context_map_v1.json'
    CONTEXT_MODE = 'text'  # 'code' or 'text'
    OUTPUT_PATH = f"model/sbert_ce_{CONTEXT_MODE}"
    train_sbert_st(data_path=DATA_PATH, 
                   context_codes_path=CONTEXT_CODES_PATH,
                   context_mode=CONTEXT_MODE,
                   output_path=OUTPUT_PATH)