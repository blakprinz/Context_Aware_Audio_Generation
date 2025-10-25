from ultralytics import YOLO
import json

def detect_objects(image_path):
    """    Detects objects in an image using your custom YOLO model and returns structured results.
    """
    # 1️⃣ Load your trained custom model (update path if needed)
    model = YOLO("runs/detect/train/weights/best.pt")
    # 2️⃣ Run YOLO inference on the image
    results = model(image_path)
    # 3️⃣ Prepare the output list
    detections = []
    # 4️⃣ Loop through all detected boxes
    for box in results[0].boxes:
        cls_id = int(box.cls)                     # class index (0, 1, 2, ...)
        label = model.names[cls_id]               # class name from your YAML
        conf = float(box.conf)                    # confidence score
        x1, y1, x2, y2 = box.xyxy[0].tolist()     # bounding box (top-left & bottom-right)

        detections.append({
            "label": label,
            "bbox": (x1, y1, x2, y2),
            "confidence": conf
        })
    # 5️⃣ Print results neatly
    print("\n🧾 Detected Objects:")
    for d in detections:
        print(f"- {d['label']} ({d['confidence']*100:.1f}%) at {d['bbox']}")
    # 6️⃣ (Optional) Save results to a JSON file
    output_file = "detections.json"
    with open(output_file, "w") as f:
        json.dump(detections, f, indent=4)
    print(f"\n✅ Detection results saved to: {output_file}")
    # 7️⃣ Return detections for further processing (e.g., scene graph)
    return detections
# ------------------- #
# 🔹 Run the function #
# ------------------- #
if __name__ == "__main__":
    image_path = r"path\test_image.jpg"
    detect_objects(image_path)