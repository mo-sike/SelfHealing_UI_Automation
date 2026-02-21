import numpy as np
from sklearn.neighbors import NearestNeighbors


class UIGraphBuilder:

    def build_graph(self, detections, k=3):

        if len(detections) == 0:
            return {}

        # Compute center coordinates
        centers = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            centers.append([cx, cy])

        centers = np.array(centers)

        # KNN based graph
        nbrs = NearestNeighbors(n_neighbors=min(k, len(centers))).fit(centers)
        distances, indices = nbrs.kneighbors(centers)

        graph = {}

        for i, neighbors in enumerate(indices):
            # Remove self-loop
            neighbor_list = [n for n in neighbors if n != i]
            graph[i] = neighbor_list

        return graph
