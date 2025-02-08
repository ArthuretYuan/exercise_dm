import torch
from sklearn.metrics import classification_report
import numpy as np
from exercise_dm.core.training.train_concat_bert import Dataset, bert_tokenizer, quatization_label, DistilBertConcatMultiPassageClassifier


# Load test data
data_df = bert_tokenizer(input_data_path='data/test_data.csv')

# Load the model
model = torch.load('model/concat_bert/concat_bert_model.pt')

model.eval()
test_data = Dataset(data_df)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=4)
device = "mps"

kpi_save_path = 'model/concat_bert/kpi.txt'
y_true_test, y_pred_test = [], []
total_diff = 0
with torch.no_grad():
    print('start testing ...')
    for test_input, test_label in test_dataloader:
        test_label = test_label.to(device)
        mask = test_input['attention_mask'].to(device)
        input_id = test_input['input_ids'].to(device)
        output = torch.squeeze(model(input_id, mask))
        
        for pred_output, true_score in zip(output,test_label):
            pred_score = 1/(1 + np.exp(-pred_output.to('cpu').detach().numpy()))
            y_pred_test.append(quatization_label(pred_score))
            y_true_test.append(quatization_label(true_score))
            print(f"Predicted Similarity Score: {pred_score:.4f}, Actual Score: {true_score:.4f}")
            abs_diff = abs(pred_score - true_score)
            total_diff += abs_diff
            
    report = classification_report(
        np.array(y_true_test),
        np.array(y_pred_test),
        output_dict=False,
        target_names=['Unrelated', 'Somewhat related', 'Synonyms', 'Close synonyms', 'Very close'])
    

    mean_diff = total_diff / len(data_df)
    with open(kpi_save_path, 'a') as f:
        f.write(f'-------------testing preformance report-------------\n')
        f.write(f'Mean Absolute Difference: {mean_diff:.4f}\n')
        f.write(f'{report}\n')

print(report)
