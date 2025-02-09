from fastapi import FastAPI
from pydantic import BaseModel
from exercise_dm.core.training.train_concat_bert import DistilBertConcatMultiPassageClassifier # noqa
from exercise_dm.jobs.infer_concat_bert import predict_with_concat_bert
from exercise_dm.jobs.infer_sbert_ce import predict_with_sbert_cross_encoder
from exercise_dm.jobs.infer_sbert_st import predict_with_sbert_sentence_transformer
from exercise_dm.jobs.infer_gpt2 import predict_with_gpt2
from exercise_dm.jobs.load_models import model_loader

# Initialize FastAPI
app = FastAPI(title="Similarity Score API")

class InputText(BaseModel):
    model: str
    context_mode: str
    text1: str
    text2: str
    context: str
    true_similarity: float

@app.post("/predict")
def predict_similarity(input_text: InputText):
    """Predict similarity score between two texts using multiple models."""

    model = model_loader(model_name=input_text.model, context_mode=input_text.context_mode)
    print('model loaded successfully')

    if input_text.model == "concat_bert":
        pred_score = predict_with_concat_bert(model, input_text.text1, input_text.text2, input_text.context)
    elif input_text.model == "sbert_st":
        pred_score = predict_with_sbert_sentence_transformer(model, input_text.text1, input_text.text2, input_text.context)
    elif input_text.model == "sbert_ce":
        pred_score = predict_with_sbert_cross_encoder(model, input_text.text1, input_text.text2, input_text.context)
    elif input_text.model == "gpt2":
        pred_score = predict_with_gpt2(model, input_text.text1, input_text.text2, input_text.context)
    else:
        pred_score = 0.0
    return {"predicted_similarity": [float(pred_score)]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)