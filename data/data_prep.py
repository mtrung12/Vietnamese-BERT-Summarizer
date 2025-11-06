import os
import pandas as pd

BASE_PATH = 'data/VietnameseMDS'
data = []
for cluster in os.listdir(BASE_PATH):
    cluster_path = os.path.join(BASE_PATH, cluster)
    if not os.path.isdir(cluster_path):
        continue
    
    body_files = [f for f in os.listdir(cluster_path) if f.endswith('.body.txt')]
    info_files = [f for f in os.listdir(cluster_path) if f.endswith('.info.txt')]

    cluster_texts = []
    for body_file in body_files:
        body_path = os.path.join(cluster_path, body_file)
        with open(body_path, 'r', encoding='utf-8') as f:
            cluster_texts.append(f.read().strip())
    join_text = '\n'.join(cluster_texts)
    ref1_path = os.path.join(cluster_path, f"{cluster}.ref1.txt")
    ref2_path = os.path.join(cluster_path, f"{cluster}.ref2.txt")
    summary_1 = ''
    with open(ref1_path, 'r', encoding='utf-8') as f:
        summary_1 = f.read().strip()
    
    summary_2 = ''
    with open(ref2_path, 'r', encoding='utf-8') as f:
        summary_2 = f.read().strip()
    
    
    data.append({
        "cluster": cluster,
        "text": join_text,
        "human_summary_1": summary_1,
        "human_summary_2": summary_2
    })
    
df = pd.DataFrame(data)
df.to_csv('VietnameseMDS.csv')