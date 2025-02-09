# This script trains a Sentence-BERT Cross Encoder model using the 'xxx' pre-trained model

import csv
import json
from sklearn.model_selection import train_test_split
import torch
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from exercise_dm.settings import PRE_TRAINED_MODEL_SBERT_CE


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
                data.append([row[1], row[2], context_code_mapping[row[3]]])
            labels.append(float(row[4]))
    return data, labels


def train_sbert_ce(data_path, context_codes_path, context_mode, output_path):
    
    # Split the data into train, validation, and test sets (80% train, 10% validation, 10% test)
    data, labels = load_data(data_path, context_codes_path, context_mode)
    train_data, val_test_data, train_labels, val_test_labels = train_test_split(
        data, labels, test_size=0.2, random_state=42)
    val_data, _, val_labels, _ = train_test_split(
        val_test_data, val_test_labels, test_size=0.5, random_state=42)

    # Re-format the data to the compatible format for Sentence-BERT Cross Encoder
    train_samples = []
    for train_texts, score in zip(train_data, train_labels):
        train_samples.append(InputExample(texts=[f'{train_texts[0]} [SEP] {train_texts[2]}', f'{train_texts[1]} [SEP] {train_texts[2]}'], label=score))
    val_samples = []
    for val_stexts, score in zip(val_data, val_labels):
        val_samples.append(InputExample(texts=[f'{val_stexts[0]} [SEP] {val_stexts[2]}', f'{val_stexts[1]} [SEP] {val_stexts[2]}'], label=score))
    evaluator = CECorrelationEvaluator.from_input_examples(val_samples, name="sts-dev")
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=8)

    # Model selection
    # REF: https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss
    # REF: https://sbert.net/docs/cross_encoder/pretrained_models.html#stsbenchmark
    # 'cross-encoder/stsb-distilroberta-base' is a pre-trained cross encoder based on 'distill_roberta', the size is 328MB
    # 'cross-encoder/stsb-roberta-base' is a pre-trained cross encoder based on 'distill_roberta', the size is 499MB
    model = CrossEncoder(PRE_TRAINED_MODEL_SBERT_CE)

    # Train the model
    model.fit(
        output_path=output_path,
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        evaluation_steps=1000,
        epochs=15,
        optimizer_params={'lr': 5e-5},
        show_progress_bar=True,
    )

    # Save the models
    torch.save(model, f'{output_path}/sbert_ce_model.pt')


if __name__ == '__main__':
    DATA_PATH = 'data/data.csv'
    CONTEXT_CODES_PATH = 'data/code_context_map_v1.json'
    CONTEXT_MODE = 'text'  # 'code' or 'text'
    OUTPUT_PATH = f"model/sbert_ce_{CONTEXT_MODE}"
    train_sbert_ce(data_path=DATA_PATH, 
                   context_codes_path=CONTEXT_CODES_PATH,
                   context_mode=CONTEXT_MODE,
                   output_path=OUTPUT_PATH)