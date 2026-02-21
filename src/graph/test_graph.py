import sys
import os
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..")))

try:
    from src.detection.rico_parser import RICOParser
    from graph_builder import UIGraphBuilder
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)
parser = RICOParser()
graph_builder = UIGraphBuilder()

detections = parser.parse_json(
    r"C:\Workspcae\SelfHealing_UI_Automation\data\101.json")

print(f"Number of UI Controls: {len(detections)}")

graph = graph_builder.build_graph(detections)

print("\nGraph Structure:")
for node, neighbors in graph.items():
    print(f"Node {node} -> Neighbors {neighbors}")
