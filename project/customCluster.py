from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np

def customCluster(data):
    #clustering = AgglomerativeClustering(n_clusters=None, compute_full_tree=True, distance_threshold=5)
    #clustering.fit(data)
    clusterings = [KMeans(n_clusters=i) for i in range(2,7)]
    clusterings = [cluster.fit(data) for cluster in clusterings]

    scores = []
    for l,clustering in enumerate(clusterings):
        occurences = [list(clustering.labels_).count(i) for i in range(l+2)]
        score = 0
        for occurence in occurences:
            if occurence < 5:
                score += 0
            else:
                score += min([abs(occurence-i) for i in [9,12,16]])
        scores.append(score)

    return clusterings[np.argmin(scores)]
