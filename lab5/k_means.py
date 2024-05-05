import numpy as np

def initialize_centroids_forgy(data: np.ndarray, k: int):
    chosen_rows = np.random.choice(np.arange(data.shape[0]), size=k, replace=False)
    return data[chosen_rows, :]

def initialize_centroids_kmeans_pp(data: np.ndarray, k: int):
    centroid = np.empty((k, data.shape[1]))
    first_idx = np.random.randint(data.shape[0])
    centroid[0, :] = data[first_idx, :]

    for i in np.arange(1, k):
        distances = np.sqrt(np.sum((data.reshape((data.shape[0], 1, data.shape[1])) - centroid[:i, :].reshape((1, i, centroid.shape[1])))**2, axis=-1))
        sum_distances = np.sum(distances, axis=-1)
        max_idx = np.argmax(sum_distances)
        centroid[i, :] = data[max_idx, :]

    return centroid

def assign_to_cluster(data: np.ndarray, centroid: np.ndarray):
    distances = np.sum((data.reshape((data.shape[0], 1, data.shape[1])) - centroid.reshape((1, *centroid.shape)))**2, axis=-1)
    assignments: np.ndarray = np.argmin(distances, axis=-1)
    return assignments

def update_centroids(data: np.ndarray, assignments: np.ndarray, num_centroids: int):
    new_centroids = np.empty((num_centroids, data.shape[1]))
    for assigned_class in np.arange(num_centroids):
        new_centroids[assigned_class, :] = np.mean(data[assignments == assigned_class], axis=0)
    return new_centroids

def mean_intra_distance(data: np.ndarray, assignments: np.ndarray, centroids: np.ndarray):
    return np.sqrt(np.sum((data - centroids[assignments, :])**2))

def k_means(data: np.ndarray, num_centroids: int, kmeansplusplus=False):
    # centroids initizalization
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else: 
        centroids = initialize_centroids_forgy(data, num_centroids)

    assignments  = assign_to_cluster(data, centroids)
    for i in range(100): # max number of iteration = 100
        print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments, num_centroids)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments): # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)         

