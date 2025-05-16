import cv2
import numpy as np
from roboflow import Roboflow
import supervision as sv
import os
from PIL import Image

# Initialize Roboflow
rf = Roboflow(api_key="zvk9zaWxnlyi6jkKUbTR")  # Replace with your Roboflow API key
project = rf.workspace("tumbalzzz").project("fullface-helmet-helmet-no-helmet-detection")
model = project.version(1).model

def detect_helmets(image_path):
    try:
        # First check if the image exists
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return None, None

        # Load image with PIL first
        pil_image = Image.open(image_path).convert('RGB')
        # Convert to numpy array for OpenCV
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Run inference on the image
        result = model.predict(image_path, confidence=40, overlap=30).json()
        
        # Parse predictions
        detections = []
        for prediction in result['predictions']:
            x, y, width, height = prediction['x'], prediction['y'], prediction['width'], prediction['height']
            
            # Convert to xyxy format (required by supervision)
            x1 = x - width/2
            y1 = y - height/2
            x2 = x + width/2
            y2 = y + height/2
            
            confidence = prediction['confidence']
            class_name = prediction['class']
            class_id = prediction.get('class_id', 0)  # Default to 0 if not present
            
            detections.append([x1, y1, x2, y2, confidence, class_id, class_name])
        
        # Create supervision detections object
        if detections:
            # Create a simple box annotator with custom colors
            box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.from_hex(['#FF5733', '#33FF57', '#3357FF']))
            
            # Extract necessary components for annotation
            xyxy = np.array([d[:4] for d in detections])
            confidences = np.array([d[4] for d in detections])
            class_ids = np.array([d[5] for d in detections], dtype=int)
            class_names = [d[6] for d in detections]
            
            # Create labels
            labels = [f"{class_name}: {confidence:.2f}" 
                     for class_name, confidence in zip(class_names, confidences)]
            
            # Annotate the image
            annotated_image = box_annotator.annotate(
                scene=image.copy(),
                detections=sv.Detections(
                    xyxy=xyxy,
                    confidence=confidences,
                    class_id=class_ids
                ),
                labels=labels
            )
            
            # Add detection count overlay
            detection_text = f"Detected: {len(detections)} helmet(s)"
            cv2.putText(annotated_image, detection_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display results with details
            print(f"\nDetected {len(detections)} objects:")
            for i, det in enumerate(detections):
                print(f"  {i+1}. {det[6]}, Confidence: {det[4]:.2f}")
            
            # Display the image
            cv2.imshow('Helmet Detection', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Save annotated image
            output_path = f"output_{os.path.basename(image_path)}"
            cv2.imwrite(output_path, annotated_image)
            print(f"\nSaved annotated image to {output_path}")
            
            return detections, annotated_image
        else:
            print("No helmets detected")
            cv2.imshow('Helmet Detection', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return [], image

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None, None

if __name__ == "__main__":
    # Example usage for a single image
    image_path = input("Enter the path to your image (or press Enter for default image): ").strip()
    if not image_path:
        # Use a default image from the project directory
        image_path = "1.jpeg"
    
    if not os.path.isabs(image_path):
        # Convert relative path to absolute path
        image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), image_path)
    
    detect_helmets(image_path)