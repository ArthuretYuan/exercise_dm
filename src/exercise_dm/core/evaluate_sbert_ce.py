import csv
import torch
from sklearn.metrics import classification_report
import numpy as np


# Load test data
data, labels = [], []
with open('data/test_data.csv', newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(csvreader)  # Skip the header (id,main,target,context,score)
    for row in csvreader:
        data.append([row[1], row[2], row[3]])
        labels.append(float(row[4]))

# Load the model
model = torch.load('model/sbert_ce/sbert_ce_model.pt')

total_diff = 0
y_pred_val, y_true_val = [], []
for texts, label in zip(data, labels):
    print(texts, label)
    # Format input (same as during training)
    # Define expressions and context
    main_expr = texts[0]
    target_expr = texts[1]
    context_code = texts[2]
    sentences = [(f"{main_expr} [SEP] {context_code}", f"{target_expr} [SEP] {context_code}")]


    # Predict similarity score
    similarity_score = model.predict(sentences)[0]


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


mean_diff = total_diff / len(data)

report = classification_report(
    np.array(y_true_val),
    np.array(y_pred_val),
    output_dict=False,
    target_names=['Unrelated', 'Somewhat related', 'Synonyms', 'Close synonyms', 'Very close'])
print(report)


kpi_save_path = 'model/sbert_ce/kpi.txt'
with open(kpi_save_path, 'a') as f:
    f.write(f'-------------testing preformance report-------------\n')
    f.write(f'Mean Absolute Difference: {mean_diff:.4f}\n')
    f.write(f'{report}\n')