import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# IMPORTANT: Update these paths to match your project structure
MODEL_PATH = 'models/best_model.keras'
IMAGE_PATH = 'test_samples/val/XR_ELBOW_negative_1788.png' # Change this to your test image

# IMPORTANT: This MUST match the image size your model was trained on!
# Check your training script. It's likely (416, 416) or (512, 512).
IMAGE_SIZE = (416, 416)

# The class names from your training script
CLASS_NAMES = [
    'XR_ELBOW_positive', 'XR_FINGER_positive', 'XR_FOREARM_positive',
    'XR_HAND_positive', 'XR_SHOULDER_positive',
    'XR_ELBOW_negative', 'XR_FINGER_negative', 'XR_FOREARM_negative',
    'XR_HAND_negative', 'XR_SHOULDER_negative'
]
CONFIDENCE_THRESHOLD = 0.5 # Only show predictions with confidence > 50%


def preprocess_image(img_path):
    """Loads and preprocesses an image for model prediction."""
    # Load the original image
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Could not find or open the image at: {img_path}")

    # Store the original image for visualization later
    original_image = image.copy()
    
    # Resize and normalize the image for the model
    image_resized = cv2.resize(image, IMAGE_SIZE)
    image_normalized = image_resized / 255.0
    
    # Add a batch dimension because the model expects it
    image_batch = np.expand_dims(image_normalized, axis=0)
    
    return original_image, image_batch

def draw_predictions(image, predictions):
    """Draws bounding boxes and labels on the image."""
    # Unpack the model's outputs and remove the extra "batch" dimension
    class_pred = predictions[0][0]  # Shape changes from (1, 10) to (10,)
    bbox_pred = predictions[1][0]    # Shape changes from (1, 10, 4) to (10, 4)
    # obj_pred = predictions[2][0]   # Not used in this function, but this is how you'd get it
    
    # Get the original image dimensions
    h, w, _ = image.shape
    
    # Find the class with the highest probability
    predicted_class_id = np.argmax(class_pred)
    confidence = class_pred[predicted_class_id]
    
    print(f"Predicted Class ID: {predicted_class_id}, Name: {CLASS_NAMES[predicted_class_id]}")
    print(f"Confidence: {confidence:.2f}")

    if confidence > CONFIDENCE_THRESHOLD:
        # Get the bounding box for the predicted class
        # Bbox format from model: [x_center, y_center, width, height] (normalized)
        box = bbox_pred[predicted_class_id]
        
        # Convert normalized coordinates to pixel coordinates
        center_x = int(box[0] * w)
        center_y = int(box[1] * h)
        box_width = int(box[2] * w)
        box_height = int(box[3] * h)
        
        x_min = int(center_x - box_width / 2)
        y_min = int(center_y - box_height / 2)
        
        # Draw the bounding box
        cv2.rectangle(image, (x_min, y_min), (x_min + box_width, y_min + box_height), (0, 255, 0), 2)
        
        # Create the label text
        label = f"{CLASS_NAMES[predicted_class_id]}: {confidence:.2f}"
        
        # Draw the label background
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x_min, y_min - text_height - 10), (x_min + text_width, y_min), (0, 255, 0), -1)
        
        # Put the label text
        cv2.putText(image, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
    return image

def main():
    """Main function to run the model testing."""
    # 1. Load the trained model
    print(f"Loading model from: {MODEL_PATH}")
    # The 'compile=False' can speed up loading for inference-only tasks
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully!")

    # 2. Load and preprocess the image
    original_image, image_for_model = preprocess_image(IMAGE_PATH)

    # 3. Make a prediction
    print("Running prediction...")
    # The model.predict() returns a list of outputs, in the order they are defined in the Model
    predictions = model.predict(image_for_model)

    # 4. Draw predictions on the original image
    output_image = draw_predictions(original_image, predictions)

    # 5. Display the result
    # Convert BGR (OpenCV's default) to RGB (Matplotlib's default)
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title("Model Prediction")
    plt.axis('off') # Hide axes
    plt.show()


if __name__ == "__main__":
    main()