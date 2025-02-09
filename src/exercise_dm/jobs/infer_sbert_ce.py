# Predict Simliarity with sbert cross encoder
  
def predict_with_sbert_cross_encoder(model, text1, text2, context):
    sentences = [(f"{text1} [SEP] {context}", f"{text2} [SEP] {context}")]
    pred_score = model.predict(sentences)[0]

    return pred_score