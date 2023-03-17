import numpy as np
import cv2
from PIL import Image
import torch
from transformers import AutoImageProcessor
from transformers import AutoModelForImageClassification

print("We are using PyTorch version " + torch.__version__)

model_dir = "/tmp/vit-base-custom-dataset"
image     = Image.open('Image.jpg').convert('RGB') 

image_processor = AutoImageProcessor.from_pretrained(model_dir)
inputs          = image_processor(image, return_tensors="pt")
model           = AutoModelForImageClassification.from_pretrained(model_dir)
with torch.no_grad():
    logits = model(**inputs).logits
    predicted_label = logits.argmax(-1).item()
    print(model.config.id2label[predicted_label])