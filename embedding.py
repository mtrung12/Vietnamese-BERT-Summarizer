import numpy as np
from transformers import AutoTokenizer, AutoModel, BertTokenizer,  BertModel
import torch
from config import VNCORENLP_PATH
from vnnlpcore import mvn_word_tokenize

def get_sentence_embeddings(sentences, model_name, device):
    if "vibert4news" in model_name.lower():
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)  
    elif "phobert" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModel.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
    model.to(device)
    model.eval()
    
    embeddings = []
    batch_size = 32
    for i in range(0, len(sentences), batch_size):
        batch_sents = sentences[i:i+batch_size]
        if "phobert" in model_name.lower():
            batch_sents = [ mvn_word_tokenize(sent) for sent in batch_sents]
        inputs = tokenizer(batch_sents, padding=True, truncation=False, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings.extend(cls_embeddings.cpu().numpy())
    
    return np.array(embeddings)