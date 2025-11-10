import numpy as np
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
import torch
from config import VNCORENLP_PATH
from vnnlpcore import mvn_word_tokenize
MAX_LENGTH = 256

def get_sentence_embeddings(sentences, model_name, device):
    if "vibert4news" in model_name.lower():
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        MAX_LENGTH = 512
    elif "phobert" in model_name.lower():
        MAX_LENGTH = 256
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModel.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        MAX_LENGTH = 512

        
    model.to(device)
    model.eval()
    
    embeddings = []
    batch_size = 32
    valid_sentences = []

    for sent in sentences:
        if "phobert" in model_name.lower():
            sent = mvn_word_tokenize(sent)
        tokens = tokenizer.encode(sent, add_special_tokens=True)
        if len(tokens) > MAX_LENGTH:
            print(f"WARNING: Skipping long sentence ({len(tokens)} tokens > {MAX_LENGTH}): {sent[:100]}...")
            continue
        valid_sentences.append(sent)

    for i in range(0, len(valid_sentences), batch_size):
        batch_sents = valid_sentences[i:i+batch_size]
        inputs = tokenizer(batch_sents, padding=True, truncation=False, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings.extend(cls_embeddings.cpu().numpy())
    
    return np.array(embeddings)