import argparse
from tqdm import tqdm
import numpy as np
import os
import json
from config import MODEL_MAP, DEVICE
from data_loader import load_and_preprocess_data
from embedding import get_sentence_embeddings
from summarizer import generate_summary
from evaluate import evaluate_rouge
import pandas as pd

def main(model_name):
    parser = argparse.ArgumentParser(description="Vietnamese Extractive MDS with BERT + KMeans")
    parser.add_argument("--model_name", type = str, required = True)
    parser.add_argument("--data_dir", type = str, required = False, default = 'data/VietnameseMDS.csv' , help = "Data files in .csv with following columns cluster(id of the record), text, human_summary_i (i = 1,2,3,...)")
    parser.add_argument("--output_dir", type = str, required = False, default = 'result', help = 'Directory path to save result')
    parser.add_argument("--compression_rate", "-r", type = float, required = False, default = 0.4, help = "Rate of length of output to length of input based on number of syllables")
    args = parser.parse_args()
    if args.model_name not in MODEL_MAP:
        raise ValueError(f"Model name {args.model_name} not supported. Available: {list(MODEL_MAP.keys())}")
    
    full_model_name = MODEL_MAP[args.model_name]
    print(f"\nProcessing model: {full_model_name} ({args.model_name})")
    clusters = load_and_preprocess_data(args.data_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    csv_file = os.path.join(args.output_dir, f"{args.model_name}_csv.csv")
    
    detailed_results = []
    r1_scores = []
    r2_scores = []
    
    for cluster in tqdm(clusters):
        sentences = cluster['sentences']
        input_text = " ".join(sentences)
        embeddings = get_sentence_embeddings(sentences, full_model_name, DEVICE)
        generated_sum, output_syllables, input_syllables = generate_summary(sentences, embeddings, args.compression_rate)
        r1, r2 = evaluate_rouge(generated_sum, cluster['human_sums'])
        r1_scores.append(r1)
        r2_scores.append(r2)
        detailed_results.append({
            "input_text": input_text,
            "generated_sum": generated_sum,
            "compression_rate": args.compression_rate,
            "input_syllables": input_syllables,
            "output_syllables": output_syllables,
            "rouge_1": round(r1 * 100, 2),
            "rouge_2": round(r2 * 100, 2)
        })
    
    avg_r1 = np.mean(r1_scores) * 100
    avg_r2 = np.mean(r2_scores) * 100
    print(f"Average ROUGE-1: {avg_r1:.2f}%")
    print(f"Average ROUGE-2: {avg_r2:.2f}%")
    
    results_dir = args.output_dir
    os.makedirs(results_dir, exist_ok=True) 
    result_file = os.path.join(results_dir, f"{model_name}_rate{args.compression_rate}_res.json")
    results = {
        "model": model_name,
        "full_model_name": full_model_name,
        "avg_rouge_1": avg_r1,
        "avg_rouge_2": avg_r2
    }
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {result_file}")
    
    df_new = pd.DataFrame(detailed_results)
    if os.path.exists(csv_file):
        df_old = pd.read_csv(csv_file)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        df_combined.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"Appended {len(df_new)} new rows to existing {csv_file}")
    else:
        df_new.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"Saved detailed summaries to {csv_file}")

    print(f"Detailed CSV saved to {csv_file}")

if __name__ == "__main__":
   main()