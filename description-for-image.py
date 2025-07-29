# !pip install -q transformers googletrans==4.0.0-rc1
# !pip install -q arabic_reshaper python-bidi

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from googletrans import Translator
from google.colab import files
import matplotlib.pyplot as plt

# Upload your image
uploaded = files.upload()
image_path = next(iter(uploaded))
img = Image.open(image_path).convert("RGB")

# Load BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Generate English caption
inputs = processor(images=img, return_tensors="pt")
out = model.generate(**inputs)
english_caption = processor.decode(out[0], skip_special_tokens=True)



translator = Translator()
translated = translator.translate(english_caption, src='en', dest='fa')
persian_caption = translated.text

# برای پشتیبانی از فارسی
import arabic_reshaper
from bidi.algorithm import get_display

reshaped_text = arabic_reshaper.reshape(persian_caption)
bidi_persian_caption = get_display(reshaped_text)

# Display image with English caption
plt.figure(figsize=(8,6))
plt.imshow(img)
plt.title(english_caption+'\n'+bidi_persian_caption)
plt.axis('off')
plt.show()
