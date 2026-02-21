import numpy as np


def visual_similarity(boxA, boxB):
    # Simple placeholder: compare box sizes
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return 1 - abs(areaA - areaB) / max(areaA, areaB)


def match_nodes(boxesA, boxesB, threshold=0.5):
    matches = []

    for i, boxA in enumerate(boxesA):
        best_score = 0
        best_j = -1

        for j, boxB in enumerate(boxesB):
            score = visual_similarity(boxA, boxB)

            if score > best_score:
                best_score = score
                best_j = j

        if best_score >= threshold:
            matches.append((i, best_j, best_score))
        else:
            matches.append((i, None, best_score))

    return matches
