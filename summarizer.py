import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

def count_syllable(text):
    return len(text.strip().split())

def generate_summary(sentences, embeddings, compression_rate=0.3):
    if not sentences or len(sentences) == 0:
        return ""
    
    n_sentences = len(sentences)
    
    # Compute total input syllables
    total_syllables = sum(count_syllable(s) for s in sentences)
    if total_syllables == 0:
        return sentences[0] if sentences else ""
    
    max_allowed_syllables = int(total_syllables * compression_rate)
    if max_allowed_syllables < 10:
        max_allowed_syllables = max(10, total_syllables // 10)  # minimum reasonable output
    
    print(f"Input: {n_sentences} sentences, {total_syllables} syllables")
    print(f"Hard limit: ≤ {max_allowed_syllables} syllables (rate={compression_rate:.2%})")

    # Start with a reasonable k = max_allowed_syllables / avg_syllables_per_sent
    avg_syllables_per_sent = total_syllables / n_sentences
    initial_k = max(1, min(n_sentences, round(max_allowed_syllables / avg_syllables_per_sent) + 1)) # add 1 here just for the case summarizer choose too short sentences --> might miss info

    # Run KMeans to get centroids 
    kmeans = KMeans(n_clusters=initial_k, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(embeddings)
    centroids = kmeans.cluster_centers_
    distances = euclidean_distances(embeddings, centroids)
    closest_indices = np.argmin(distances, axis=0)  # one per cluster

    # Rank candidates by distance to centroid
    candidates = []
    for cluster_idx, sent_idx in enumerate(closest_indices):
        sent = sentences[sent_idx]
        dist = distances[sent_idx, cluster_idx]
        syl = count_syllable(sent)
        candidates.append({
            'idx': sent_idx,
            'sent': sent,
            'dist': dist,
            'syllables': syl
        })
    
    # Sort by importance (smaller distance = more important)
    candidates.sort(key=lambda x: x['dist'])

    # Greedy selection: add until we can't without exceeding
    selected = []
    current_syllables = 0
    exceeded_options = []  # store ones that would exceed (for fallback)

    for cand in candidates:
        if current_syllables + cand['syllables'] <= max_allowed_syllables:
            selected.append(cand)
            current_syllables += cand['syllables']
        else:
            exceeded_options.append(cand)

    # If under budget and have room, add smallest exceeding one
    if selected and exceeded_options:
        exceeded_options.sort(key=lambda x: x['syllables'])  # smallest first
        for cand in exceeded_options:
            if current_syllables + cand['syllables'] <= max_allowed_syllables + 10:  # allow +10 syl tolerance
                selected.append(cand)
                current_syllables += cand['syllables']
                print(f"  Added smallest exceeding sentence (+{cand['syllables']} syl, total now {current_syllables})")
                break

    # Sort selected sentences by original order 
    selected.sort(key=lambda x: x['idx'])
    summary_sents = [item['sent'] for item in selected]
    summary = " ".join(summary_sents)
    final_syllables = count_syllable(summary)

    achieved_rate = final_syllables / total_syllables

    print(f"Output: {len(summary_sents)} sentences, {final_syllables} syllables "
          f"({achieved_rate:.1%} of input){' [EXCEEDED BY {}]'.format(final_syllables - max_allowed_syllables) if over else ''}")

    return summary, final_syllables, total_syllables