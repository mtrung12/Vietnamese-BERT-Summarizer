import numpy as np
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
import torch
from config import VNCORENLP_PATH
from vnnlpcore import mvn_word_tokenize

def get_sentence_embeddings(sentences, model_name, device):
    if "vibert4news" in model_name.lower():
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        max_length = 512 
    elif "phobert" in model_name.lower():
        max_length = 256
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModel.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        max_length = 512

    model.to(device)
    model.eval()

    # Collect all input dicts for batches
    all_batch = []
    sentence_indices = []  # List of (start, end) for chunk indices per sentence
    current_chunk_idx = 0

    for sent_idx, sent in enumerate(sentences):
        if "phobert" in model_name.lower():
            sent = mvn_word_tokenize(sent)

        # Get input_ids without special tokens
        input_ids = tokenizer.encode(sent, add_special_tokens=False)

        if len(input_ids) == 0:
            continue

        num_chunks = 0
        if len(input_ids) + 2 > max_length:
            # Chunking for long sentences
            stride = 128
            max_chunk_len = max_length - 2  # Room for [CLS] and [SEP]
            start = 0
            while start < len(input_ids):
                end = min(start + max_chunk_len, len(input_ids))
                chunk_ids = input_ids[start:end]
                # Prepare dict without padding or tensor
                input_dict = tokenizer.prepare_for_model(
                    chunk_ids,
                    max_length=max_length,
                    padding=False,
                    truncation=True,
                    return_tensors=None
                )
                all_batch.append(input_dict)
                num_chunks += 1
                if end == len(input_ids):
                    break
                start += stride
        else:
            # Normal short sentence
            input_dict = tokenizer.prepare_for_model(
                input_ids,
                max_length=max_length,
                padding=False,
                truncation=True,
                return_tensors=None
            )
            all_batch.append(input_dict)
            num_chunks = 1

        if num_chunks > 0:
            sentence_indices.append((current_chunk_idx, current_chunk_idx + num_chunks))
            current_chunk_idx += num_chunks
        else:
            print(f"WARNING: Empty chunks for sentence: {sent[:100]}...")

    # Now process in batches
    batch_size = 32
    chunk_embeddings = []
    for i in range(0, len(all_batch), batch_size):
        batch = all_batch[i:i + batch_size]
        # Pad the batch to the longest in the batch
        inputs = tokenizer.pad(batch, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        chunk_embeddings.extend(cls_embeddings)

    # Now average per sentence
    sentence_embs = []
    for start, end in sentence_indices:
        if start < end:
            avg_emb = np.mean(chunk_embeddings[start:end], axis=0)
            sentence_embs.append(avg_emb)

    return np.array(sentence_embs)