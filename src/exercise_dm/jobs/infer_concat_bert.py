import torch
import numpy as np
from transformers import BertTokenizer
from exercise_dm.jobs.load_models import DEVICE 


# Load the model
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

def predict_with_concat_bert(model, text1, text2, context):
    model.eval()
    input_elements = [text1, text2, context]
    token = tokenizer(input_elements, padding='max_length', max_length = 512, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        mask = token['attention_mask'].to(DEVICE)
        input_id = token['input_ids'].to(DEVICE)
        output = torch.squeeze(model(input_id.unsqueeze(0), mask.unsqueeze(0)))
        pred_score = 1/(1 + np.exp(-output.to('cpu').detach().numpy()))

    return pred_score