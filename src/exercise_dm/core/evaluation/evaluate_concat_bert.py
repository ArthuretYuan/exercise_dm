import torch
from sklearn.metrics import classification_report
import numpy as np
from exercise_dm.core.training.train_concat_bert import Dataset, bert_tokenizer, DistilBertConcatMultiPassageClassifier
from exercise_dm.utils.quantize_similarity_score import quatization_label
from exercise_dm.utils.measure_false_degree import measure_false_degree


def evaluate_concat_bert(model_path, test_data_path, context_codes_path, context_mode, kpi_save_path):
    
    # Load test data
    data_df = bert_tokenizer(data_path=test_data_path, context_codes_path=context_codes_path, context_mode=context_mode)
    test_data = Dataset(data_df)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=4)

    # Load the model
    model = torch.load(model_path, map_location=torch.device('mps'))
    model.eval()
    
    # evaluate the model
    device = "mps"
    y_true_test, y_pred_test = [], []
    total_diff = 0
    with torch.no_grad():

        # Format input (same as during training)
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].to(device)
            
            # Predict similarity score
            output = torch.squeeze(model(input_id, mask))
            for pred_output, true_score in zip(output,test_label):
                pred_score = 1/(1 + np.exp(-pred_output.to('cpu').detach().numpy()))
                
                # Quantize the similarity score
                y_pred_test.append(quatization_label(pred_score))
                y_true_test.append(quatization_label(true_score))
                print(f"Predicted Similarity Score: {pred_score:.4f}, Actual Score: {true_score:.4f}")
                abs_diff = abs(pred_score - true_score)
                total_diff += abs_diff

        # calculate the classification report
        report = classification_report(
            np.array(y_true_test),
            np.array(y_pred_test),
            output_dict=False,
            target_names=['Unrelated', 'Somewhat related', 'Synonyms', 'Close synonyms', 'Very close'])
        
        # calculate the MSE
        mean_diff = total_diff / len(data_df)
        
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
    print(report)


if __name__ == '__main__':
    CONTEXT_MODE = 'code'
    MODEL_PATH = f'model/concat_bert_{CONTEXT_MODE}/concat_bert_model.pt'
    TEST_DATA_PATH = 'data/test_data.csv'
    CONTEXT_CODES_PATH = 'data/code_context_map_v1.json'
    KPI_SAVE_PATH = f'model/concat_bert_{CONTEXT_MODE}/testing_kpi.txt'
    evaluate_concat_bert(
        model_path=MODEL_PATH,
        test_data_path=TEST_DATA_PATH,
        context_codes_path=CONTEXT_CODES_PATH,
        context_mode=CONTEXT_MODE,
        kpi_save_path=KPI_SAVE_PATH
    )
