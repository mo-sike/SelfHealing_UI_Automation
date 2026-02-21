import json


class RICOParser:

    def parse_json(self, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)

        detections = []
        self._extract_nodes(data, detections)
        return detections

    def _extract_nodes(self, node, detections):

        # Only consider clickable elements or meaningful components
        clickable = node.get("clickable", False)
        component_label = node.get("componentLabel", "")
        bounds = node.get("bounds", None)

        if bounds and clickable:
            x1, y1, x2, y2 = bounds

            if x2 > x1 and y2 > y1:
                detections.append({
                    "bbox": bounds,
                    "label": node.get("class", "unknown"),
                    "text": node.get("text", "")
                })

        for child in node.get("children", []):
            self._extract_nodes(child, detections)
