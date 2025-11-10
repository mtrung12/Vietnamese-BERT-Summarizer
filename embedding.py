import numpy as np
from transformers import AutoTokenizer, AutoModel, BertTokenizer, AutoModelForMaskedLM, BertModel
import torch
import py_vncorenlp
from config import VNCORENLP_PATH

def get_sentence_embeddings(sentences, model_name, device):
    # for PhoBERT
    py_vncorenlp.download_model(save_dir=VNCORENLP_PATH)
    # Load the word and sentence segmentation component
    segmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=VNCORENLP_PATH)
    
    if "vibert4news" in model_name.lower():
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)  
    elif "phobert" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
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
            batch_sents = [
                " ".join(segmenter.word_segment(sent))   # e.g. "Học_sinh được nghỉ_học"
                for sent in batch_sents
            ]
        inputs = tokenizer(batch_sents, padding=True, truncation=False, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings.extend(cls_embeddings.cpu().numpy())
    
    return np.array(embeddings)