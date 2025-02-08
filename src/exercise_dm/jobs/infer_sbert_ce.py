import torch

# Load the model
model = torch.load('model/sbert_ce/sbert_ce_model.pt')

def predict_with_sbert_cross_encoder(text1, text2, context):
    sentences = [(f"{text1} [SEP] {context}", f"{text2} [SEP] {context}")]
    pred_score = model.predict(sentences)[0]

    return pred_score