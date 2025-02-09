import torch
from transformers import GPT2Tokenizer
from exercise_dm.jobs.load_models import DEVICE 

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Use <|endoftext|> as padding

# tokenizer
def encode_sentences(data, tokenizer):
    inputs = [f"{texts[0]} <|sep|> {texts[1]} <|sep|> {texts[2]}" for texts in data]
    return tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")

def predict_with_gpt2(model, text1, text2, context):
    texts = [text1, text2, context]
    encodings = encode_sentences([texts], tokenizer).to(DEVICE)
    with torch.no_grad():
        pre_score = model(**encodings).logits[0][0]
    pre_score = max(0, min(1, pre_score))
    return pre_score