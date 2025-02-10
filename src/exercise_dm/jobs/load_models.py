import torch
from exercise_dm.core.training.train_concat_bert import DistilBertConcatMultiPassageClassifier #noqa

FINETUNED_MODEL_SBERT_ST_CODE = "model/sbert_st_code/sbert_st_model.pt"
FINETUNED_MODEL_SBERT_ST_TEXT = "model/sbert_st_text/sbert_st_model.pt"
FINETUNED_MODEL_SBERT_CE_CODE = "model/sbert_ce_code/sbert_ce_model.pt"
FINETUNED_MODEL_SBERT_CE_TEXT = "model/sbert_ce_text/sbert_ce_model.pt"
FINETUNED_MODEL_CONCAT_BERT_CODE = "model/concat_bert_code/concat_bert_model.pt"
FINETUNED_MODEL_CONCAT_BERT_TEXT = "model/concat_bert_text/concat_bert_model.pt"
FINETUNED_MODEL_GPT2_CODE = "model/gpt2_code/gpt2_model.pt"
FINETUNED_MODEL_GPT2_TEXT = "model/gpt2_text/gpt2_model.pt"

if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "mps"

def model_loader(model_name, context_mode):
    model = None
    if context_mode == "code":
        if model_name == "sbert_st":
            model = torch.load(FINETUNED_MODEL_SBERT_ST_CODE, map_location=torch.device(DEVICE))
        elif model_name == "sbert_ce":
            model = torch.load(FINETUNED_MODEL_SBERT_CE_CODE, map_location=torch.device(DEVICE))
        elif model_name == "concat_bert":
            model = torch.load(FINETUNED_MODEL_CONCAT_BERT_CODE, map_location=torch.device(DEVICE))
        elif model_name == "gpt2":
            model = torch.load(FINETUNED_MODEL_GPT2_CODE, map_location=torch.device(DEVICE))
    elif context_mode == "text":
        if model_name == "sbert_st":
            model = torch.load(FINETUNED_MODEL_SBERT_ST_TEXT, map_location=torch.device(DEVICE))
        elif model_name == "sbert_ce":
            model = torch.load(FINETUNED_MODEL_SBERT_CE_TEXT, map_location=torch.device(DEVICE))
        elif model_name == "concat_bert":
            model = torch.load(FINETUNED_MODEL_CONCAT_BERT_TEXT, map_location=torch.device(DEVICE))
        elif model_name == "gpt2":
            model = torch.load(FINETUNED_MODEL_GPT2_TEXT, map_location=torch.device(DEVICE))
    return model