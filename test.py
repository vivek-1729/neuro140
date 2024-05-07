import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from gensim.models import Word2Vec
from sklearn.cluster import AgglomerativeClustering
import pickle


with open("agenda", "rb") as file:
    data = pickle.load(file)

data = [word for sublist in data for word in sublist]

categories = list(set(data))

# Tokenize categories into words
tokenized_categories = [category.split() for category in categories]

# Set up parameters for Word2Vec training
embedding_dim = 100  # Adjust as needed
window_size = 5  # Adjust as needed
min_count = 1  # Adjust as needed

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=tokenized_categories, vector_size=embedding_dim, window=window_size,
                          min_count=min_count)

# Function to get the embedding vector for a single category
def get_embedding(category):
    words = category.split()  # Split category into words
    embedding = np.zeros(word2vec_model.vector_size)  # Initialize embedding vector
    for word in words:
        if word in word2vec_model.wv:
            embedding += word2vec_model.wv[word]
    return embedding / len(words)  # Return average of word embeddings

# Convert categories to embeddings
category_embeddings = np.array([get_embedding(category) for category in categories])

# Define number of clusters (adjust as needed)
num_clusters = 10

# Apply agglomerative clustering
clustering = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='ward')
cluster_labels = clustering.fit_predict(category_embeddings)

# Function to plot dendrogram
def plot_dendrogram(model, **kwargs):
    plt.figure(figsize=(20, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    dendrogram(model, leaf_rotation=90, leaf_font_size=8, **kwargs)
    plt.savefig("dendrogram.png")
    plt.show()

# Convert clustering labels to linkage matrix for dendrogram
from scipy.cluster.hierarchy import linkage
linkage_matrix = linkage(category_embeddings, method='ward')
plot_dendrogram(linkage_matrix, labels=cluster_labels, distance_sort='descending')

# Print the top categories in each cluster
for i in range(num_clusters):
    print(f"Cluster {i}:")
    cluster_indices = np.where(cluster_labels == i)[0]
    cluster_categories = [categories[idx] for idx in cluster_indices]
    print(cluster_categories[:10])  # Print top 10 categories in the cluster

from sklearn.manifold import MDS

# Perform multidimensional scaling (MDS) to reduce dimensionality to 2D
mds = MDS(n_components=2, dissimilarity="euclidean", random_state=42)
embedding_2d = mds.fit_transform(category_embeddings)

# Plot clusters
plt.figure(figsize=(10, 8))
for i in range(num_clusters):
    cluster_indices = np.where(cluster_labels == i)[0]
    cluster_embeddings = embedding_2d[cluster_indices]
    plt.scatter(cluster_embeddings[:, 0], cluster_embeddings[:, 1], label=f'Cluster {i}')

plt.title('Clusters Visualization (2D)')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
plt.savefig("visualization.png")
plt.show()