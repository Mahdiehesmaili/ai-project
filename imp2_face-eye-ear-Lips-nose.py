# !pip install mediapipe opencv-python
# !pip install face_recognition
# !pip install cmake
# !pip install dlib
# !pip install opencv-python

import cv2
import mediapipe as mp
from google.colab.patches import cv2_imshow
from google.colab import files
import numpy as np

# Upload image
uploaded = files.upload()
image_path = next(iter(uploaded))

# Load the image
image = cv2.imread(image_path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    results = face_detection.process(rgb_image)

    # Draw detections
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(image, detection)
    else:
        print("No faces detected.")

# Show result
cv2_imshow(image)

# Optional: Save the result
output_path = "faces_detected.jpg"
cv2.imwrite(output_path, image)
print(f"Saved to {output_path}")
