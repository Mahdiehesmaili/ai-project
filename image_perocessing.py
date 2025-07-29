from google.colab import files
from PIL import Image
from IPython.display import display

# Upload image
uploaded = files.upload()
image_path = next(iter(uploaded))
img = Image.open(image_path)
display(img)

!pip install opencv-python-headless

import cv2
import matplotlib.pyplot as plt

# Load image
image = cv2.imread(image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)  # Enhance contrast


# Load Haar cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detect faces
###faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.05,     # More sensitive to smaller changes
    minNeighbors=3,       # Lower value = more detections
    minSize=(30, 30)      # Minimum object size to detect
)
# Draw rectangles
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Show the result
plt.figure(figsize=(8,6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title(f"Detected Faces: {len(faces)}")
plt.axis('off')
plt.show()
cv2.imwrite("detected_faces.jpg", image)
files.download("detected_faces.jpg")
