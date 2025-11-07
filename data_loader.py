import pandas as pd
from underthesea import sent_tokenize, word_tokenize
from config import DATA_PATH

def load_and_preprocess_data():
    df = pd.read_csv(DATA_PATH)
    clusters = []
    for _, row in df.iterrows():
        cluster_id = row['cluster']
        raw_text = row['text']
        sentences = sent_tokenize(raw_text)
        human_sum1 = row['human_summary_1']
        human_sum2 = row['human_summary_2']
        human_sums = [word_tokenize(h, format="text") for h in [human_sum1, human_sum2]]
        clusters.append({
            'cluster_id': cluster_id,
            'sentences': sentences,
            'human_sums': human_sums
        })
    return clusters