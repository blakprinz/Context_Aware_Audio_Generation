import torch
import numpy as np
from ultralytics import YOLO

# This is the master list of all 256 objects. 
OBJECT_LIST = [
    'Phone', 'Keys', 'Wallet', 'ID card', 'Watch', 'Headphones', 'Laptop', 'Mouse',
    'Keyboard', 'Monitor', 'Desk', 'Chair', 'Pen', 'Pencil', 'Notebook', 'Book',
    'Coffee mug', 'Water bottle', 'Plate', 'Fork', 'Spoon', 'Knife', 'Bowl', 'Cup',
    'Napkin', 'Paper towel', 'Trash can', 'Light switch', 'Lamp', 'Couch', 'Television',
    'Remote control', 'Speaker', 'Router', 'Power outlet', 'Charger', 'USB cable',
    'Backpack', 'Shopping bag', 'Car', 'Bicycle', 'Helmet', 'Jacket', 'T-shirt',
    'Jeans', 'Shoes', 'Socks', 'Underwear', 'Belt', 'Hat', 'Gloves', 'Umbrella',
    'Mirror', 'Hairbrush', 'Toothbrush', 'Toothpaste', 'Soap', 'Shampoo',
    'Conditioner', 'Towel', 'Toilet paper', 'Faucet', 'Sink', 'Toilet', 'Showerhead',
    'Razor', 'Shaving cream', 'Deodorant', 'Perfume', 'Hairspray', 'Comb', 'Band-aid',
    'Medication bottle', 'Thermometer', 'First-aid kit', 'Tissue box', 'Eyeglasses',
    'Sunglasses', 'Contact lenses', 'Credit card', 'Cash', 'Coins', 'Checkbook',
    'Envelope', 'Stamp', 'Postcard', 'Calendar', 'Clock', 'Alarm clock',
    'Picture frame', 'Vase', 'Plant', 'Flowerpot', 'Curtains', 'Blinds', 'Pillow',
    'Blanket', 'Sheet', 'Mattress', 'Bed frame', 'Wardrobe', 'Hanger', 'Drawer',
    'Screwdriver', 'Hammer', 'Pliers', 'Tape measure', 'Flashlight', 'Battery',
    'Lighter', 'Matches', 'Candle', 'Magazine', 'Newspaper', 'Mail', 'Bills',
    'Receipt', 'Scissors', 'Glue', 'Tape', 'Stapler', 'Paperclip', 'Rubber band',
    'Folder', 'Binder', 'Marker', 'Highlighter', 'Eraser', 'Ruler', 'Compass',
    'Protractor', 'Calculator', 'Whiteboard', 'Dry-erase marker', 'Chalk', 'Sponge',
    'Dish soap', 'Dishwasher', 'Dish rack', 'Oven', 'Microwave', 'Toaster',
    'Blender', 'Mixer', 'Coffee maker', 'Kettle', 'Can opener', 'Corkscrew',
    'Spatula', 'Whisk', 'Tongs', 'Cutting board', 'Pot', 'Pan', 'Lid', 'Oven mitts',
    'Apron', 'Refrigerator', 'Freezer', 'Ice cube tray', 'Jar', 'Container',
    'Tupperware', 'Food storage bag', 'Aluminum foil', 'Plastic wrap', 'Garbage bag',
    'Broom', 'Dustpan', 'Vacuum cleaner', 'Mop', 'Bucket', 'Cleaning spray', 'Paper',
    'Cardboard box', 'Cereal box', 'Milk carton', 'Juice bottle', 'Soda can',
    'Wine bottle', 'Beer bottle', 'Bottle opener', 'Wine glass', 'Drinking glass',
    'Coaster', 'Placemat', 'Salt shaker', 'Pepper shaker', 'Spice rack', 'Grill',
    'Barbecue tongs', 'Garden hose', 'Shovel', 'Rake', 'Watering can', 'Sprinkler',
    'Lawnmower', 'Scented candle', 'Air freshener', 'Coat hanger', 'Iron',
    'Ironing board', 'Laundry basket', 'Clothesline', 'Dryer sheets', 'Detergent',
    'Fabric softener', 'Bleach', 'Stain remover', 'Sewing kit', 'Thread', 'Needle',
    'Safety pin', 'Button', 'Zipper', 'Elastic band', 'Rubber gloves',
    'Hand sanitizer', 'Face mask', 'Shopping cart', 'Shopping basket',
    'Vending machine', 'ATM', 'Ticket', 'Passport', 'Luggage', 'Suitcase', 'Map',
    'Binoculars', 'Camera', 'Tripod', 'Microphone', 'Game controller',
    'Playing cards', 'Dice', 'Jigsaw puzzle', 'Lego bricks', 'Doll', 'Toy car',
    'Ball', 'Jump rope', 'Skateboard', 'Scooter', 'Swing', 'Slide',
    'Teeter-totter', 'Sandbox', 'Beach ball', 'Beach towel', 'Sunscreen'
]

# Create a mapping from class name to its index for quick lookups
CLASS_TO_INDEX = {name: i for i, name in enumerate(OBJECT_LIST)}

def generate_context_vector(image_path: str, model_path: str) -> np.ndarray:
    """
    Takes an image path and a trained model path, detects objects,
    and returns a 256-dimensional binary vector.

    Args:
        image_path (str): The path to the input image.
        model_path (str): The path to the fine-tuned YOLO model (.pt file).

    Returns:
        np.ndarray: A 1D numpy array of length 256 with 0s and 1s.
    """
    # Initialize the vector with all zeros
    context_vector = np.zeros(256, dtype=int)

    # Load the fine-tuned model
    model = YOLO(model_path)

    results = model(image_path, verbose=False) # verbose=False to suppress output

    for r in results:

        detected_class_ids = r.boxes.cls.int().tolist()

        detected_class_names = {model.names[i] for i in detected_class_ids}

        for name in detected_class_names:
            if name in CLASS_TO_INDEX:
                index = CLASS_TO_INDEX[name]
                context_vector[index] = 1

    return context_vector

def main():
    # --- IMPORTANT ---
    # Replace this with the actual path to your fine-tuned model's weights
    TRAINED_MODEL_PATH = 'runs/custom_yolo_training/weights/best.pt'
    
    # Replace this with the path to the image you want to process
    IMAGE_TO_PROCESS = 'path/to/your/test_image.jpg'

    print(f"Processing image: {IMAGE_TO_PROCESS}")
    
    try:
        # Generate the vector
        output_vector = generate_context_vector(IMAGE_TO_PROCESS, TRAINED_MODEL_PATH)
        
        print("\nGenerated Context Vector (1D array of 256 elements):")
        print(output_vector)
        
        # Optional: Print which objects were detected
        detected_objects = [OBJECT_LIST[i] for i, val in enumerate(output_vector) if val == 1]
        print("\nDetected Objects:")
        if detected_objects:
            for obj in detected_objects:
                print(f"- {obj}")
        else:
            print("No objects from the list were detected.")

    except FileNotFoundError:
        print(f"Error: Make sure the model path '{TRAINED_MODEL_PATH}' and image path '{IMAGE_TO_PROCESS}' are correct.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()