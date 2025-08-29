from ultralytics import YOLO

def main():
    # Load a pre-trained YOLO model.
    model = YOLO('yolov8n.pt')

    # Fine-tune the model on your custom dataset
    # The 'data' argument points to your YAML file.
    print("Starting model fine-tuning...")
    results = model.train(
        data='data.yaml',
        epochs=100,  
        imgsz=640,
        project='runs', # Directory to save results
        name='custom_yolo_training' # Sub-directory name for this specific run
    )
    print("Fine-tuning complete!")
    print("Trained model and results are saved in the 'runs/custom_yolo_training' directory.")

if __name__ == '__main__':
    main()