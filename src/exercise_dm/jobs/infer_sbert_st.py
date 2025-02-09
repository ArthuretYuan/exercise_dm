# Predict similarity with sbert sentence transformer

def predict_with_sbert_sentence_transformer(model, text1, text2, context):
    sentences = [f"{text1} [SEP] {context}", f"{text2} [SEP] {context}"]
    embeddings = model.encode(sentences, convert_to_tensor=True)
    pred_score = model.similarity(embeddings, embeddings)[0][1]

    return pred_score