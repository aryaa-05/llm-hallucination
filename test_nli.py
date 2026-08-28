import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def test_nli(model_name, claim, evidence):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    print(f"Model: {model_name}")
    print("id2label:", model.config.id2label)
    
    enc = tokenizer(evidence, claim, return_tensors="pt", max_length=256, truncation=True)
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0]
    
    print("Logits:", logits)
    print("Probs:", probs)
    print("---")

claim = "She won the Nobel Prize in Physics in 1903."
evidence = "Marie Curie shared the Nobel Prize in Physics in 1903 with Pierre Curie and Henri Becquerel."

test_nli("cross-encoder/nli-deberta-v3-small", claim, evidence)
test_nli("cross-encoder/nli-deberta-v3-base", claim, evidence)
