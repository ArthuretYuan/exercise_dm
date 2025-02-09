import csv
import json
import torch
from transformers import GPT2Tokenizer
from sklearn.metrics import classification_report
import numpy as np


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Use <|endoftext|> as padding

# tokenizer
def encode_sentences(data, tokenizer):
    inputs = [f"{texts[0]} <|sep|> {texts[1]} <|sep|> {texts[2]}" for texts in data]
    return tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")

# load context code mapping
with open('data/code_context_map_v1.json', 'r') as f:
    context_code_mapping = json.load(f)


# load model
model = torch.load('model/gpt2/gpt2_model.pt', map_location=torch.device('mps'))
model.eval()

with open('data/test_data.csv', newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(csvreader)  # Skip the header (id,main,target,context,score)

    total_diff = 0
    y_pred_val, y_true_val = [], []
    size = 0
    for row in csvreader:
        size += 1
        data = [[row[1], row[2], context_code_mapping[row[3]].lower()]]
        label = float(row[4])

        encodings = encode_sentences(data, tokenizer).to('mps')

        with torch.no_grad():
            similarity_score = model(**encodings).logits[0][0]
        similarity_score = max(0, min(1, similarity_score))

        # Normalize score (if needed)
        # normalized_score = (similarity_score + 1) / 2  

        print(f"Predicted Similarity Score: {similarity_score:.4f}, Actual Score: {label:.4f}")
        abs_diff = abs(similarity_score - label)
        total_diff += abs_diff

        #print(f"Normalized Similarity Score: {normalized_score:.4f}")


        match similarity_score:
            case num if num <  0.125:
                y_pred_val.append(0) # Unrelated
            case num if 0.125 <= num < 0.375:
                y_pred_val.append(1) # Somewhat related
            case num if 0.375 <= num < 0.625:
                y_pred_val.append(2) # Synonyms
            case num if 0.625 <= num < 0.875:
                y_pred_val.append(3) # Close synonyms
            case num if num >= 0.875:
                y_pred_val.append(4) # Very close
        
        match label:
            case num if num <  0.125:
                y_true_val.append(0) # Unrelated
            case num if 0.125 <= num < 0.375:
                y_true_val.append(1) # Somewhat related
            case num if 0.375 <= num < 0.625:
                y_true_val.append(2) # Synonyms
            case num if 0.625 <= num < 0.875:
                y_true_val.append(3) # Close synonyms
            case num if num >= 0.875:
                y_true_val.append(4) # Very close


    mean_diff = total_diff / size

    report = classification_report(
        np.array(y_true_val),
        np.array(y_pred_val),
        output_dict=False,
        target_names=['Unrelated', 'Somewhat related', 'Synonyms', 'Close synonyms', 'Very close'])
    print(report)


    kpi_save_path = 'model/gpt2/kpi.txt'
    with open(kpi_save_path, 'a') as f:
        f.write(f'-------------testing preformance report-------------\n')
        f.write(f'Mean Absolute Difference: {mean_diff:.4f}\n')
        f.write(f'{report}\n')

