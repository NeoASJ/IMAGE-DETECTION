import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os


IMG_SIZE = (48, 48)  # Must be the same size used for training
MODEL_PATH = 'best_emotion_model.keras'
SAMPLE_IMAGE_PATH = r'DATASET\train\fear\Training_12567.jpg'# <<-- CHANGE THIS TO YOUR IMAGE FILE


CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES = len(CLASS_NAMES)


print(f"Loading model from: {MODEL_PATH}")
# Load the saved model
try:
    loaded_model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure you have run the training code and the model file exists.")
    # Exit if model load fails
    exit()

# Display the model summary to confirm
loaded_model.summary()
def prepare_single_image(img_path):
    """
    Loads, resizes, normalizes, and reshapes a single image for prediction.
    """
    if not os.path.exists(img_path):
        print(f"Error: Image file not found at {img_path}")
        return None

    # Load the image
    img = image.load_img(img_path, target_size=IMG_SIZE)
    
    # Convert image to numpy array
    img_array = image.img_to_array(img)
    
   
    img_array = np.expand_dims(img_array, axis=0)
    
    
    normalized_array = img_array / 255.0
    
    print(f"Image loaded and prepared with shape: {normalized_array.shape}")
    return normalized_array


sample_tensor = prepare_single_image(SAMPLE_IMAGE_PATH)

if sample_tensor is None:
    
    exit()
# --- Make the prediction ---
print("\nMaking Prediction...")
predictions = loaded_model.predict(sample_tensor)

# The result is an array of probabilities (one-hot encoded)
print(f"Raw Prediction Output: {predictions}")

# Find the index of the highest probability
predicted_class_index = np.argmax(predictions[0])

# Get the corresponding class name
predicted_class_name = CLASS_NAMES[predicted_class_index]

# Get the confidence level
confidence = np.max(predictions[0]) * 100

# --- Print Final Result ---
print("--- Prediction Result ---")
print(f"The model predicts the emotion is: **{predicted_class_name.upper()}**")
print(f"Confidence: {confidence:.2f}%")
print("-------------------------")