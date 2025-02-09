import csv
import json
import torch
import numpy as np
from sklearn.metrics import classification_report
from exercise_dm.utils.quantize_similarity_score import quatization_label
from exercise_dm.utils.measure_false_degree import measure_false_degree


def load_test_data(test_data_path, context_codes_path, context_mode):

    # load context code mapping
    with open(context_codes_path, 'r') as f:
        context_code_mapping = json.load(f)

    # load data and labels
    data, labels = [], []
    with open(test_data_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(csvreader)  # Skip the header (id,main,target,context,score)
        for row in csvreader:
            if context_mode == 'code':
                data.append([row[1], row[2], row[3]])
            elif context_mode == 'text':
                data.append([row[1], row[2], context_code_mapping[row[3]]])
            labels.append(float(row[4]))
    return data, labels


def evaluate_sbert_st(model_path, test_data_path, context_codes_path, context_mode, kpi_save_path):

    # Load the data
    data, labels = load_test_data(test_data_path, context_codes_path, context_mode)

    # Load the model
    model = torch.load(model_path, map_location=torch.device('mps'))

    total_diff = 0
    y_pred_test, y_true_test = [], []
    for texts, label in zip(data, labels):
        
        # Format input (same as during training)
        main_expr = texts[0]
        target_expr = texts[1]
        context_code = texts[2]
        sentences = [(f"{main_expr} [SEP] {context_code}", f"{target_expr} [SEP] {context_code}")]

        # Predict similarity score
        similarity_score = model.predict(sentences)[0]
        print(f"Predicted Similarity Score: {similarity_score:.4f}, Actual Score: {label:.4f}")
        abs_diff = abs(similarity_score - label)
        total_diff += abs_diff

        # Quantize the similarity score
        quatized_pred_label = quatization_label(similarity_score)
        quantized_true_label = quatization_label(label)
        y_pred_test.append(quatized_pred_label)
        y_true_test.append(quantized_true_label)

    # calculate the MSE
    mean_diff = total_diff / len(data)

    # calculate the classification report
    report = classification_report(
        np.array(y_true_test),
        np.array(y_pred_test),
        output_dict=False,
        target_names=['Unrelated', 'Somewhat related', 'Synonyms', 'Close synonyms', 'Very close'])
    print(report)

    # calculate false prediction degree
    correct_count, incorrect_count, false_degree = measure_false_degree(y_pred_test, y_true_test)
    false_degree_text = ""
    for k, v in false_degree.items():
        false_degree_text += f'\t{k}: {v}\n'
    false_degree_report = f'- Correct Count: {correct_count}\n- Incorrect Count: {incorrect_count}\n- False Degree:\n{false_degree_text}'
    
    # save the evaluation result
    with open(kpi_save_path, 'w') as f:
        f.write(f'------------- Testing Preformance Report -------------\n\n')
        f.write(f'- Mean Absolute Error: {mean_diff:.4f}\n\n')
        f.write(f'- Basic Evaluation Metrics:\n{report}\n\n')
        f.write(f'{false_degree_report}\n\n')


if __name__ == '__main__':
    CONTEXT_MODE = 'code'
    MODEL_PATH = f'model/sbert_ce_{CONTEXT_MODE}/sbert_ce_model.pt'
    TEST_DATA_PATH = 'data/test_data.csv'
    CONTEXT_CODES_PATH = 'data/code_context_map_v1.json'
    KPI_SAVE_PATH = f'model/sbert_ce_{CONTEXT_MODE}/testing_kpi.txt'
    evaluate_sbert_st(
        model_path=MODEL_PATH,
        test_data_path=TEST_DATA_PATH,
        context_codes_path=CONTEXT_CODES_PATH,
        context_mode=CONTEXT_MODE,
        kpi_save_path=KPI_SAVE_PATH
    )