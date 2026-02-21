import numpy as np
from sklearn.neighbors import NearestNeighbors


def build_graph(boxes, k=3):
    """
    boxes: list of bounding boxes [[x1,y1,x2,y2], ...]
    k: number of nearest neighbors
    """

    # Compute center points
    centers = []
    for box in boxes:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        centers.append([cx, cy])

    centers = np.array(centers)

    # Find K nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=min(k, len(centers))).fit(centers)
    distances, indices = nbrs.kneighbors(centers)

    graph = {}
    for i, neighbors in enumerate(indices):
        graph[i] = neighbors.tolist()

    return graph
