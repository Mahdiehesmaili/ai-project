!pip install deepface
!pip install opencv-python

fr!pip install deepface
!pip install opencv-python
om google.colab import files
import os

uploaded = files.upload()
print("Uploaded files:")
for file in uploaded.keys():
    print(file)



import cv2
from deepface import DeepFace
from matplotlib import pyplot as plt

# List your known face images with labels
known_faces = [
    {"name": "nyayesh", "img": "nyayesh.jpg"},
    {"name": "amir", "img": "amir.jpg"}
]

# Test group photo
group_image_path = file#"image1.jpg"
group_img = cv2.imread(group_image_path)

# Analyze the image to find faces
detected_faces = DeepFace.extract_faces(img_path=group_image_path, enforce_detection=False)




# Loop over detected faces
for face in detected_faces:
    # x, y, w, h = face["facial_area"].values()
    facial_area = face["facial_area"]
    x = facial_area["x"]
    y = facial_area["y"]
    w = facial_area["w"]
    h = facial_area["h"]

    # Extract face region
    detected_face = group_img[y:y+h, x:x+w]

    label = "Unknown"
    for person in known_faces:
        result = DeepFace.verify(img1_path=person["img"], img2_path=detected_face, enforce_detection=False)
        if result["verified"]:
            label = person["name"]
            break

    # Draw rectangle and label
    cv2.rectangle(group_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(group_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

# Show final image
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(group_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
