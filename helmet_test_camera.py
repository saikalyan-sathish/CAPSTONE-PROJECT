import cv2
import numpy as np
import tensorflow as tf
import os
import sys
import glob

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
model_path = '/Users/saikalyansathish/Desktop/CAPSTONE PROJECT 2/capstone_model'
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

print("Model setup complete. Starting webcam...")

# Define class names based on your training setup
class_names = ['helmet', 'half-helmet', 'no helmet']

# Define preprocessing function to match your training pipeline
def preprocess_frame(frame):
    # Resize to 224x224 (model input size)
    frame = cv2.resize(frame, (224, 224))
    # Convert BGR to RGB (TensorFlow models expect RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert to float32 and normalize to [0, 1]
    frame = frame.astype(np.float32) / 255.0
    # Normalize with mean and std used during training
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    frame = (frame - mean) / std
    # Expand dimensions to add batch axis (1, 224, 224, 3)
    frame = np.expand_dims(frame, axis=0)
    return frame

# Safe prediction function with error handling
def safe_predict(model, preprocessed_frame):
    """Make a prediction with error handling and confidence threshold"""
    try:
        predictions = model.predict(preprocessed_frame, verbose=0)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][predicted_class] * 100
        
        # Add confidence threshold
        CONFIDENCE_THRESHOLD = 60  # Adjust this value based on your needs (60%)
        if confidence < CONFIDENCE_THRESHOLD:
            return 0, confidence  # Default to 'helmet' if not confident
            
        return predicted_class, confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return 0, 0.0  # Return default values

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 is usually the built-in webcam

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit(1)

print("\n--- INSTRUCTIONS ---")
print("Press 'q' to quit")
print("Press 's' to save the current frame")
print("-------------------\n")

try:
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam")
            break

        # Create a copy for display
        display_frame = frame.copy()

        # Preprocess the frame
        preprocessed = preprocess_frame(frame)

        # Make prediction
        predicted_class, confidence = safe_predict(model, preprocessed)

        # Create label
        label = f"{class_names[predicted_class]} ({confidence:.2f}%)"

        # Add text to the frame
        cv2.putText(
            display_frame,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0) if predicted_class == 0 else (0, 165, 255) if predicted_class == 1 else (0, 0, 255),
            2
        )

        # Display the frame
        cv2.imshow("Helmet Detection (Webcam)", display_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('s'):
            # Save the current frame
            filename = f"helmet_detection_{len(glob.glob('helmet_detection_*.jpg')) + 1}.jpg"
            cv2.imwrite(filename, display_frame)
            print(f"Saved frame as {filename}")

except Exception as e:
    print(f"Error during webcam processing: {str(e)}")

finally:
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")
