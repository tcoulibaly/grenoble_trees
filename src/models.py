import sklearn
import pandas as pd
import pytest
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from automl_alex.data_prepare import *

RANDOM_SEED = 42

from sklearn.metrics import silhouette_score

def get_silhouette_score(c, data):
    clusterer = KMeans(n_clusters=c, n_init="auto", random_state=10)
    cluster_labels = clusterer.fit_predict(data)
    return silhouette_score(data, cluster_labels)

def plot_silhouette_score(data, range_max):
    clusters = []
    silhouette_scores = []
    range_max = range_max+1
    for c in range(2,range_max):
        clusters.append(c)
        silhouette_scores.append(get_silhouette_score(c, data))
    df_silhouette = pd.DataFrame(
    {'clusters': clusters,
     'Silhouette scores': silhouette_scores
    })
    df_silhouette = df_silhouette.set_index('clusters')
    df_silhouette.plot()
    plt.xlabel('Clusters')
    plt.ylabel('Silhouette scores')
    plt.title('Sihouette Score for KMeans Clustering')
    plt.legend()
    plt.show()


from yellowbrick.cluster import KElbowVisualizer

km = KMeans(random_state=42)
visualizer = KElbowVisualizer(km, k=(2,10))
 
visualizer.fit(fat_quantity_scaled[deaths_obesity], 10)
visualizer.show()

