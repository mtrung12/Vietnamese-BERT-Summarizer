import pandas as pd
from underthesea import sent_tokenize         
from vncorenlp import VnCoreNLP                
import config



def mvn_word_tokenize(text: str) -> str:
    tokenized = mvncorenlp.tokenize(text)          # list[list[str]]
    flat = [token for sent in tokenized for token in sent]
    return " ".join(flat)


def load_and_preprocess_data(data_path: str):
    mvncorenlp = VnCoreNLP(
    config.VNCORENLP_PATH,
    annotators="wseg",         
    max_heap_size='-Xmx2g'
    )
    df = pd.read_csv(data_path)
    clusters = []

    for _, row in df.iterrows():
        cluster_id = row['cluster']
        raw_text   = row['text']
        sentences = sent_tokenize(raw_text)
        human_sums = [
            mvn_word_tokenize(value)                    
            for key, value in row.items()
            if key.startswith("human_summary_")
        ]

        clusters.append({
            'cluster_id': cluster_id,
            'sentences'  : sentences,
            'human_sums' : human_sums
        })

    return clusters
