from detect_objects import detect_objects
from build_scene_graph import build_scene_graph
import json

def main():
    image_path = r"image path"

    print("🔍 Detecting objects...")
    detections = detect_objects(image_path)

    print("🧠 Building scene graph...")
    scene_graph = build_scene_graph(detections)
    print(json.dumps(scene_graph, indent=2))

if __name__ == "__main__":
    main()
