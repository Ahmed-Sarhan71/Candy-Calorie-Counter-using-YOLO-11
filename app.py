import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os

st.set_page_config(page_title="Candy Calorie Counter", layout="wide")
st.title("Candy Calorie Counter")

nutrition_info = {'TuC': [146, 1.2], 'HoHos': [380, 8], 'Borio': [250, 10], 'McVities': [220, 18], 'Biskrem': [120, 18]}

model_path = st.sidebar.text_input("Model path", "my_model.pt")
conf_thresh = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5)

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

option = st.radio("Choose an image", ["Upload your own"] + sample_labels)

if option == "Upload your own":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        st.stop()
    image = Image.open(uploaded_file)
else:
    idx = sample_labels.index(option)
    image = Image.open(sample_images[idx])
    st.image(image, caption="Original image", use_container_width=True)

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

    if conf > conf_thresh:
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        label = f'{classname}: {int(conf*100)}%'
        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        candies_detected.append(classname)

total_calories = 0
total_sugar = 0
for candy_name in candies_detected:
    calories, sugar = nutrition_info[candy_name]
    total_calories += calories
    total_sugar += sugar

col1, col2 = st.columns(2)
with col1:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame_rgb, caption="Detection results", use_container_width=True)
with col2:
    st.subheader("Results")
    st.write(f"Candies detected: {len(candies_detected)}")
    st.write(f"Total calories: {total_calories}")
    st.write(f"Total sugar: {total_sugar}g")
    st.divider()
    st.subheader("Per candy")
    for candy in set(candies_detected):
        count = candies_detected.count(candy)
        cal, sug = nutrition_info[candy]
        st.write(f"{candy}: {count}x ({cal} cal, {sug}g sugar each)")
