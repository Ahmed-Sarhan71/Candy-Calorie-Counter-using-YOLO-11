import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageOps
import os

st.set_page_config(page_title="Candy Calorie Counter", layout="wide")

st.markdown("""
# Candy Calorie Counter
Upload a photo of your candies and this app will detect them, count them, and tally up total calories and sugar.
""")

nutrition_info = {'TuC': [146, 1.2], 'HoHos': [380, 8], 'Borio': [250, 10], 'McVities': [220, 18], 'Biskrem': [120, 18]}

model_path = "my_model.pt"
if not os.path.exists(model_path):
    st.error(f"Model not found at {model_path}")
    st.stop()

with st.spinner("Loading model..."):
    model = YOLO(model_path, task='detect')
labels = model.names

img_dir = "images for testing"
os.makedirs(img_dir, exist_ok=True)
sample_images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.upper().endswith(('.JPG', '.JPEG', '.PNG'))]
sample_images.sort()
sample_labels = [os.path.basename(f) for f in sample_images]

option = st.selectbox("Choose a sample image", sample_labels)
idx = sample_labels.index(option)
image = Image.open(sample_images[idx])

if image.width < image.height:
    image = image.rotate(270, expand=True)

original = image.copy()
frame = np.array(image)
frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

results = model(frame, verbose=False)
detections = results[0].boxes

candies_detected = []
for i in range(len(detections)):
    xyxy_tensor = detections[i].xyxy.cpu()
    xyxy = xyxy_tensor.numpy().squeeze()
    xmin, ymin, xmax, ymax = xyxy.astype(int)
    classidx = int(detections[i].cls.item())
    classname = labels[classidx]
    conf = detections[i].conf.item()

    if conf > 0.5:
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
        label = f'{classname}: {int(conf*100)}%'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 5)
        cv2.rectangle(frame, (xmin, ymin - th - 20), (xmin + tw + 20, ymin), (0, 255, 0), -1)
        cv2.putText(frame, label, (xmin + 10, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 5)
        candies_detected.append(classname)

total_calories = 0
total_sugar = 0
for candy_name in candies_detected:
    calories, sugar = nutrition_info[candy_name]
    total_calories += calories
    total_sugar += sugar

col1, col2 = st.columns(2)
with col1:
    st.image(original, caption="Original image", use_container_width=True)
with col2:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame_rgb, caption="Detection results", use_container_width=True)

st.divider()
res_col1, res_col2, res_col3 = st.columns(3)
res_col1.metric("Candies detected", len(candies_detected))
res_col2.metric("Total calories", total_calories)
res_col3.metric("Total sugar", f"{total_sugar}g")

if candies_detected:
    st.subheader("Per candy")
    for candy in set(candies_detected):
        count = candies_detected.count(candy)
        cal, sug = nutrition_info[candy]
        st.write(f"{candy}: {count}x ({cal} cal, {sug}g sugar each)")
