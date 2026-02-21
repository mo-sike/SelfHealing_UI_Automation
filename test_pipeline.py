import sys
import os
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..")))


try:
    from src.detection.rico_parser import RICOParser
    from src.graph.graph_builder import UIGraphBuilder
    from src.matching.matcher import GraphMatcher
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

parser = RICOParser()
graph_builder = UIGraphBuilder()
matcher = GraphMatcher()

# Load two RICO screens
detectionsA = parser.parse_json("data/raw/101.json")
detectionsB = parser.parse_json("data/raw/101.json")  # for now use same file

print(f"Screen A Controls: {len(detectionsA)}")
print(f"Screen B Controls: {len(detectionsB)}")

# Build graphs
graphA = graph_builder.build_graph(detectionsA)
graphB = graph_builder.build_graph(detectionsB)

# Perform matching
matches, unmatched = matcher.match(detectionsA, detectionsB)

print("\nMatches:")
for m in matches:
    print(m)

print("\nUnmatched Nodes (Changes):", unmatched)
