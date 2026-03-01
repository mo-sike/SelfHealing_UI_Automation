import numpy as np


class GraphMatcher:

    def visual_similarity(self, boxA, boxB):
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        return 1 - abs(areaA - areaB) / max(areaA, areaB)

    def text_similarity(self, textA, textB):
        if not textA or not textB:
            return 0
        return 1 if textA.strip().lower() == textB.strip().lower() else 0

    def combined_similarity(self, detA, detB):
        visual = self.visual_similarity(detA["bbox"], detB["bbox"])
        text = self.text_similarity(detA["text"], detB["text"])

        # weighted sum
        return 0.6 * visual + 0.4 * text

    # def match(self, detectionsA, detectionsB, threshold=0.5):

        matches = []
        unmatched = []

        for i, detA in enumerate(detectionsA):

            best_score = 0
            best_j = None

            for j, detB in enumerate(detectionsB):
                score = self.combined_similarity(detA, detB)

                if score > best_score:
                    best_score = score
                    best_j = j

            if best_score >= threshold:
                matches.append((i, best_j, best_score))
            else:
                unmatched.append(i)

        return matches, unmatched

    def match(self, detectionsA, detectionsB, threshold=0.5):

        matches = []
        unmatched = []
        used_B = set()

        for i, detA in enumerate(detectionsA):

            best_score = 0
            best_j = None

            for j, detB in enumerate(detectionsB):

                if j in used_B:
                    continue

                score = self.combined_similarity(detA, detB)

                if score > best_score:
                    best_score = score
                    best_j = j

            if best_score >= threshold and best_j is not None:
                matches.append((i, best_j, best_score))
                used_B.add(best_j)
            else:
                unmatched.append(i)

        return matches, unmatched
