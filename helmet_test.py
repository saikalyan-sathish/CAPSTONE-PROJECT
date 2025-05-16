import cv2
import numpy as np
import tensorflow as tf
import os
import sys
import glob

# Configure a single image path here - CHANGE THIS PATH TO TEST DIFFERENT IMAGES
IMAGE_PATH = '/Users/saikalyansathish/Desktop/CAPSTONE PROJECT 2/Helmet-ReiseMoto-9203.webp'

# Set TensorFlow compatibility settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
os.environ['TF_KERAS'] = '1'  # Use tf.keras instead of standalone Keras
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Print TensorFlow version for debugging
print(f"TensorFlow version: {tf.__version__}")

# Function to handle model loading safely
def load_model_safely(model_path):
    print(f"Attempting to load the model from {model_path}...")
    
    # First approach: Load with custom_objects and explicit options
    try:
        print("Approach 1: Loading model with custom_objects...")
        model = tf.keras.models.load_model(
            model_path, 
            compile=False,
            custom_objects=None
        )
        print("Model loaded successfully with Approach 1!")
        return model
    except Exception as e:
        print(f"Approach 1 failed: {str(e)}")
    
    # Second approach: Try loading with explicit TF 2.x settings
    try:
        print("Approach 2: Loading with explicit TensorFlow 2.x settings...")
        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            options=tf.saved_model.LoadOptions(
                experimental_io_device='/job:localhost'
            )
        )
        print("Model loaded successfully with Approach 2!")
        return model
    except Exception as e:
        print(f"Approach 2 failed: {str(e)}")
    
    # Third approach: Create a new model and try to load weights
    try:
        print("Approach 3: Creating base model and loading weights...")
        # Create a model with the same architecture as your trained model
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        predictions = tf.keras.layers.Dense(3, activation='softmax')(x)
        model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
        
        # Try to load just the weights
        try:
            model.load_weights(model_path)
            print("Weights loaded successfully!")
        except Exception as weight_err:
            print(f"Could not load weights: {weight_err}")
            print("Continuing with ImageNet pretrained weights only")
            
        return model
    except Exception as e:
        print(f"Approach 3 failed: {str(e)}")
    
    # Final fallback: Create a simple CNN model
    print("All loading approaches failed. Creating a simple fallback model...")
    print("WARNING: This model will not have your trained weights!")
    
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    return model

# Load the model
model_path = '/Users/saikalyansathish/Desktop/CAPSTONE PROJECT 2/best_helmet_model.h5'
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found!")
    print("Looking for any .h5 files in current directory...")
    h5_files = glob.glob("*.h5")
    if h5_files:
        model_path = h5_files[0]
        print(f"Found alternative model file: {model_path}")
    else:
        print("No .h5 model files found. Will use a fallback model.")

model = load_model_safely(model_path)

print("Model setup complete. Starting detection...")

# Define class names based on your training setup
class_names = ['helmet', 'half-helmet', 'no helmet']

# Define preprocessing function to match your training pipeline
def preprocess_frame(frame):
    # Resize to 224x224 (model input size)
    frame = cv2.resize(frame, (224, 224))
    # Convert to float32 and normalize to [0, 1]
    frame = frame.astype(np.float32) / 255.0
    # Normalize with mean and std used during training
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    frame = (frame - mean) / std
    # Expand dimensions to add batch axis (1, 224, 224, 3)
    frame = np.expand_dims(frame, axis=0)
    return frame

# Safe display function with timeout and error handling
def safe_display_image(window_name, image, timeout_ms=5000):
    """Display an image with error handling and timeout
    
    Args:
        window_name: Name of the window
        image: Image to display (numpy array)
        timeout_ms: Timeout in milliseconds (default: 5000)
                    Set to 0 for infinite wait
    
    Returns:
        Key code pressed or -1 if timeout
    """
    try:
        # Create named window with proper flags
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        
        # Show the image
        cv2.imshow(window_name, image)
        
        # Wait for key press with timeout
        key = cv2.waitKey(timeout_ms) & 0xFF
        return key
    except Exception as e:
        print(f"Warning: Display error: {e}")
        return ord('n')  # Return 'n' as if user pressed next

# Validate and check if image exists
def validate_image_path(image_path):
    """
    Validates if the provided image path exists
    
    Args:
        image_path: Path to the image file
    
    Returns:
        True if valid image path, False otherwise
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return False
        
    return True

# Safe prediction function with error handling
def safe_predict(model, preprocessed_image):
    """Make a prediction with error handling"""
    try:
        predictions = model.predict(preprocessed_image, verbose=0)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][predicted_class] * 100
        return predicted_class, confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return 0, 0.0  # Return default values

# Check if the configured image path exists
print(f"Using image: {IMAGE_PATH}")
if not validate_image_path(IMAGE_PATH):
    print("Please modify the IMAGE_PATH variable at the top of the script.")
    print("Exiting.")
    exit(1)

# Display instructions
print("\n--- INSTRUCTIONS ---")
print("Press any key to exit after viewing the results")
print("-------------------\n")

# Process the image
try:
    # Read the image
    print(f"Reading image: {IMAGE_PATH}")
    img = cv2.imread(IMAGE_PATH)
    
    if img is None:
        print(f"Failed to read image {IMAGE_PATH}. The file may be corrupted or in an unsupported format.")
        exit(1)
    
    # Create a copy for display
    display_img = img.copy()
    
    # Preprocess the image
    preprocessed = preprocess_frame(img)
    
    # Make prediction
    predicted_class, confidence = safe_predict(model, preprocessed)
    
    # Display the result
    label = f"{class_names[predicted_class]} ({confidence:.2f}%)"
    print(f"Prediction: {label}")
    
    # Add text to the image
    cv2.putText(
        display_img, 
        label, 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0, 255, 0) if predicted_class == 0 else (0, 165, 255) if predicted_class == 1 else (0, 0, 255), 
        2
    )
    
    # Display the image with the prediction
    print("Displaying image with prediction. Press any key to exit.")
    safe_display_image("Helmet Detection", display_img, 0)  # Wait indefinitely (0 timeout)
    
except Exception as e:
    print(f"Error processing image: {str(e)}")

# Clean up
cv2.destroyAllWindows()
print("Done.")
