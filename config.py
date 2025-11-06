# config.py
import torch

MODEL_MAP = {
    'vibert4news': "NlpHUST/vibert4news-base-cased",
    'phobert-base': "vinai/phobert-base",
    'phobert-large': "vinai/phobert-large",
    'xlm-r-large': "FacebookAI/xlm-roberta-large",
    'xlm-r-base': "FacebookAI/xlm-roberta-base",
    'distilbert-m': "distilbert/distilbert-base-multilingual-cased",
    'mbert-cased': "google-bert/bert-base-multilingual-cased",
    'mbert-uncased': "google-bert/bert-base-multilingual-uncased"
}

DATA_PATH = "data/VietnameseMDS.csv"
K_CLUSTERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")