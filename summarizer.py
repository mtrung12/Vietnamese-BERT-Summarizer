import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

def generate_summary(sentences, embeddings, k_clusters):
    if len(sentences) < k_clusters:
        return " ".join(sentences)
    
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    kmeans.fit(embeddings)
    
    centroids = kmeans.cluster_centers_
    distances = euclidean_distances(embeddings, centroids)
    
    closest_indices = np.argmin(distances, axis=0)
    
    selected_sents = [sentences[idx] for idx in sorted(set(closest_indices))]
    
    return " ".join(selected_sents)