import argparse
from tqdm import tqdm
import numpy as np
import os
import json
from config import MODEL_MAP, K_CLUSTERS, DEVICE
from data_loader import load_and_preprocess_data
from embedding import get_sentence_embeddings
from summarizer import generate_summary
from evaluate import evaluate_rouge

def main(model_name):
    if model_name not in MODEL_MAP:
        raise ValueError(f"Model name {model_name} not supported. Available: {list(MODEL_MAP.keys())}")
    
    full_model_name = MODEL_MAP[model_name]
    print(f"\nProcessing model: {full_model_name} ({model_name})")
    clusters = load_and_preprocess_data()
    r1_scores = []
    r2_scores = []
    
    for cluster in tqdm(clusters):
        sentences = cluster['sentences']
        embeddings = get_sentence_embeddings(sentences, full_model_name, DEVICE)
        generated_sum = generate_summary(sentences, embeddings, K_CLUSTERS)
        r1, r2 = evaluate_rouge(generated_sum, cluster['human_sums'])
        r1_scores.append(r1)
        r2_scores.append(r2)
    
    avg_r1 = np.mean(r1_scores) * 100
    avg_r2 = np.mean(r2_scores) * 100
    print(f"Average ROUGE-1: {avg_r1:.2f}%")
    print(f"Average ROUGE-2: {avg_r2:.2f}%")
    
    # Save results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)  # Create folder if it doesn't exist
    result_file = os.path.join(results_dir, f"{model_name}_res.json")
    results = {
        "model": model_name,
        "full_model_name": full_model_name,
        "avg_rouge_1": avg_r1,
        "avg_rouge_2": avg_r2
    }
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {result_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run summarization for a specific model using model name.")
    parser.add_argument("model_name", type=str, help="The model name to use (defined in config.py)")
    args = parser.parse_args()
    main(args.model_name)