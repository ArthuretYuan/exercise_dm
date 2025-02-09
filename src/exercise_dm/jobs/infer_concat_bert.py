import torch
import numpy as np
from transformers import BertTokenizer

# Load the model
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
device = "mps"

def predict_with_concat_bert(model, text1, text2, context):
    model.eval()
    input_elements = [text1, text2, context]
    token = tokenizer(input_elements, padding='max_length', max_length = 512, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        mask = token['attention_mask'].to(device)
        input_id = token['input_ids'].to(device)
        output = torch.squeeze(model(input_id.unsqueeze(0), mask.unsqueeze(0)))
        pred_score = 1/(1 + np.exp(-output.to('cpu').detach().numpy()))

    return pred_score


if __name__ == "__main__":
    text1 = "oxidizing enzyme"
    text2 = "use of oxidases"
    context = "C09"
    pred_score = predict_with_concat_bert(text1, text2, context)
    print(f"Predicted Similarity Score: {pred_score:.4f}")