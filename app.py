"""
🚗 Car Colour Detection & Counting (with People Count)
Developed by: [Sushanth]
Internship Project — 2025

Description:
This application detects cars and people at a traffic signal using YOLOv8.
It classifies cars based on colour (Blue or Other) using HSV analysis.

🟥 Red rectangles → Blue cars
🟦 Blue rectangles → Other cars
👥 People count displayed on screen

"""

import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# -----------------------------
# App Configuration
# -----------------------------
st.set_page_config(page_title="Car Colour & People Counter", layout="wide")
st.title("🚦 Car Colour Detection & Counting (with People Count)")
st.caption(
    "• Red rectangles = **Blue cars**  • Blue rectangles = **Non-blue cars**  • Counts shown below\n"
    "• Works on images and videos. Uses YOLOv8 for detection and HSV for car colour."
)

# -----------------------------
# Load YOLOv8 Model
# -----------------------------
@st.cache_resource
def load_model():
    # NOTE: Model is loaded from the 'models/' folder.
    # Make sure your yolov8n.pt file is stored at: models/yolov8n.pt
    model_path = "models/yolov8n.pt"
    return YOLO(model_path)

model = load_model()

# -----------------------------
# Constants
# -----------------------------
CAR_LABELS = {"car", "truck", "bus"}   # treat these as "car-like" for counting
PERSON_LABEL = "person"

# -----------------------------
# Helper Functions
# -----------------------------
def is_blue_car(bgr_roi, blue_thresh=0.08):
    """
    Determines whether the car ROI (Region of Interest) is 'blue'
    based on the percentage of blue pixels in HSV space.
    """
    if bgr_roi.size == 0:
        return False, 0.0

    roi_small = cv2.resize(bgr_roi, (160, 160), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(roi_small, cv2.COLOR_BGR2HSV)

    # Blue range in HSV
    lower_blue = np.array([90, 60, 60], dtype=np.uint8)
    upper_blue = np.array([130, 255, 255], dtype=np.uint8)

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_ratio = (mask_blue > 0).mean()

    return blue_ratio >= blue_thresh, float(blue_ratio)


def draw_box(img, xyxy, label, is_blue):
    """
    Draws bounding boxes on detected objects.
      - Red rectangle for blue cars
      - Blue rectangle for non-blue cars
    """
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    color = (0, 0, 255) if is_blue else (255, 0, 0)  # BGR colors
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, label, (x1 + 3, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def process_frame(bgr, blue_threshold):
    """
    Runs detection on a single BGR frame, draws rectangles, and returns counts.
    """
    h, w = bgr.shape[:2]
    results = model.predict(source=bgr, conf=0.35, verbose=False)[0]

    car_count = blue_car_count = other_car_count = people_count = 0
    names = model.model.names
    canvas = bgr.copy()

    for box in results.boxes:
        cls_id = int(box.cls.item())
        cls_name = names.get(cls_id, str(cls_id))
        xyxy = box.xyxy[0].tolist()

        if cls_name == PERSON_LABEL:
            people_count += 1
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 1)
            continue

        if cls_name in CAR_LABELS:
            car_count += 1
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w - 1, x2), min(h - 1, y2)
            car_roi = canvas[y1c:y2c, x1c:x2c]

            blue_yes, ratio = is_blue_car(car_roi, blue_thresh=blue_threshold)
            if blue_yes:
                draw_box(canvas, xyxy, f"Car: BLUE ({ratio:.2f})", is_blue=True)
                blue_car_count += 1
            else:
                draw_box(canvas, xyxy, "Car: OTHER", is_blue=False)
                other_car_count += 1

    return canvas, car_count, blue_car_count, other_car_count, people_count


# -----------------------------
# Streamlit UI
# -----------------------------
st.subheader("Input")
colL, colR = st.columns([1, 1])

with colL:
    mode = st.radio("Choose input type:", ["Image", "Video"], horizontal=True)

with colR:
    blue_threshold = st.slider(
        "Blue pixel fraction threshold (↑ stricter blue)",
        min_value=0.03, max_value=0.30, value=0.08, step=0.01
    )

# -----------------------------
# IMAGE INPUT MODE
# -----------------------------
if mode == "Image":
    img_file = st.file_uploader("Upload an image (jpg/png/webp)", type=["jpg", "jpeg", "png", "webp"])
    if img_file:
        st.write("**Preview (Original):**")
        st.image(img_file, use_column_width=True)

        image = np.array(Image.open(img_file).convert("RGB"))
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        out_bgr, cars, blue_cars, other_cars, people = process_frame(bgr, blue_threshold)
        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

        st.write("**Processed Output:**")
        st.image(out_rgb, use_column_width=True)

        st.subheader("Counts")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Cars", cars)
        m2.metric("Blue Cars", blue_cars)
        m3.metric("Other Cars", other_cars)
        m4.metric("People", people)

# -----------------------------
# VIDEO INPUT MODE
# -----------------------------
elif mode == "Video":
    vid_file = st.file_uploader("Upload a video (mp4/mov/avi)", type=["mp4", "mov", "avi", "mkv"])
    if vid_file:
        tpath = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(vid_file.name)[1])
        tpath.write(vid_file.read())
        tpath.flush()
        tpath.close()

        st.write("**Preview (First frame):**")
        cap = cv2.VideoCapture(tpath.name)
        ok, first = cap.read()
        if ok:
            st.image(cv2.cvtColor(first, cv2.COLOR_BGR2RGB), use_column_width=True)
        cap.release()

        run = st.button("Process Video")
        if run:
            st.write("**Processed Video (sampled frames):**")
            cap = cv2.VideoCapture(tpath.name)

            stframe = st.empty()
            total_cars = total_blue = total_other = total_people = 0
            frame_id = 0
            sample_every = 2  # skip frames for faster inference

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_id % sample_every == 0:
                    out_bgr, cars, blue_cars, other_cars, people = process_frame(frame, blue_threshold)
                    total_cars += cars
                    total_blue += blue_cars
                    total_other += other_cars
                    total_people += people
                    stframe.image(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)
                frame_id += 1

            cap.release()

            st.subheader("Aggregate Counts (sampled frames)")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Cars (sum)", total_cars)
            m2.metric("Blue Cars (sum)", total_blue)
            m3.metric("Other Cars (sum)", total_other)
            m4.metric("People (sum)", total_people)

            # Remove temp file
            try:
                os.remove(tpath.name)
            except Exception:
                pass

# -----------------------------
# Info Box
# -----------------------------
st.info(
    "Tip: If blue cars aren’t being detected reliably, lower the threshold. "
    "Lighting and camera white-balance can affect colours — tweak the slider for best results."
)

