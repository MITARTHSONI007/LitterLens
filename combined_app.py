import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile
from ultralytics import YOLO
from collections import Counter
from segment_anything import sam_model_registry, SamPredictor
from sklearn.cluster import KMeans
import torch
import base64

def set_background(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-blur: 50px;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Call this with your image URL or base64-encoded image
set_background("photo-1507525428034-b723cf961d3e.jpeg")
# ===== Load Models =====
st.session_state.setdefault("yolo_loaded", False)
st.session_state.setdefault("sam_loaded", False)

@st.cache_resource
def load_yolo_model():
    return YOLO("garbage_detection.pt")  # Garbage model

@st.cache_resource
def load_sam_model():
    sam_checkpoint = "sam_vit_b_01ec64.pth"
    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    sam.to("cuda" if torch.cuda.is_available() else "cpu")
    return SamPredictor(sam)


# ===== Mode 2: Water Color Detection =====
def classify_color(rgb):
    import colorsys
    r, g, b = [x / 255.0 for x in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h *= 360

    # Classify muddy brown (dark, low brightness, yellowish hue)
    if 20 <= h <= 60 and v < 0.5:
        return "muddy brown"

    if h < 30 or h > 330:
        return "red"
    elif 30 <= h <= 65:
        return "yellow"
    elif 65 < h <= 170:
        return "green"
    elif s < 0.2 and v < 0.5:
        return "grayish / unclear"
    else:
        return "unknown"



def show_color_advisory(rgb_color):
    color_category = classify_color(rgb_color)

    st.markdown(f"### 💧 Detected Water Color Category: **`{color_category.title()}`**")

    color_advisories = {
        "red": "🚨 **Unusual red color detected.** Could be from natural or industrial sources. **Extreme caution** is advised while handling this water.",
        "yellow": "⚠️ **Unusual yellow color detected.** Possible industrial or organic matter contamination. Handle with care.",
        "muddy brown": "💧 **High turbidity and suspended solids detected.** May indicate **microbial contamination** or runoff pollution.",
        "green": "🟢 **Green color detected.** Possible **eutrophication** due to high nitrogen or phosphorus levels.",
    }

    if color_category in color_advisories:
        st.warning(color_advisories[color_category])
    else:
        st.info("ℹ️ No specific advisory for this color.")
# ===== Streamlit UI =====
st.set_page_config(page_title="🌿 Environmental Analyzer", layout="wide")

# Stylish Heading
st.markdown("<h1 style='text-align: center; color: teal;'> Environmental Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Detect Garbage or Analyze Water Color with AI</h4>", unsafe_allow_html=True)

mode = st.radio("Choose Task", ["Garbage Detection", "Water Color Detection"], horizontal=True)
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# ===== Mode 1: Garbage Detection =====
st.markdown("""
    <style>
    div[data-baseweb="slider"] > div {
        background: linear-gradient(to left, #79adb3 0%, #e2c3a8 100%) !important;
    }
    </style>
    """, unsafe_allow_html=True)


if mode == "Garbage Detection":
    conf_threshold = st.slider("🎯 Confidence Threshold", min_value=0.1, max_value=0.9, value=0.3, step=0.05,)

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", width=400)

        if st.button("🚀 Detect Garbage"):
            yolo_model = load_yolo_model()
            image = Image.open(uploaded_file).convert("RGB")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                image.save(tmp.name)
                results = yolo_model(tmp.name, conf=conf_threshold)[0]

            img = np.array(image).copy()
            detected_classes = []

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{yolo_model.names[cls]} {conf:.2f}"
                detected_classes.append(yolo_model.names[cls])

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            st.image(img, caption="🧾 Detection Result", use_container_width=True)

            if detected_classes:
                st.markdown("### ♻️ Detected Garbage Types")
                count = Counter(detected_classes)
                for garbage_type, qty in count.items():
                    st.write(f"• **{garbage_type.capitalize()}**: {qty}")
            else:
                st.info("No garbage types detected.")

# ===== Mode 2 UI =====
else:
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", width=400)

        if st.button("Detect Water Color"):
            # Load models
            yolo_model = YOLO("water_detection.pt")
            predictor = load_sam_model()

            # Convert image
            image = Image.open(uploaded_file).convert("RGB")
            image_np = np.array(image)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                image.save(tmp.name)
                results = yolo_model(tmp.name)[0]

            predictor.set_image(image_np)
            found = False

            for box in results.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                input_box = np.array([x1, y1, x2, y2])
                masks, scores, _ = predictor.predict(box=input_box[None, :], multimask_output=False)

                mask = masks[0]
                segmented_pixels = image_bgr[mask]

                if segmented_pixels.size == 0:
                    continue

                k = 3
                pixels = segmented_pixels.astype(np.float32)
                brightness = np.mean(pixels, axis=1)
                filtered_pixels = pixels[brightness < 80]

                if len(filtered_pixels) < k:
                    continue

                kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                kmeans.fit(filtered_pixels)
                cluster_centers = kmeans.cluster_centers_
                labels, counts = np.unique(kmeans.labels_, return_counts=True)
                dominant_color_bgr = cluster_centers[labels[np.argmax(counts)]].astype(int)
                dominant_color_rgb = tuple(int(c) for c in dominant_color_bgr[::-1])  # BGR to RGB
                found = True

                st.markdown(f"### 🌊 Dominant Water Color (RGB): `{dominant_color_rgb}`")

                overlay = image_bgr.copy()
                overlay[mask] = dominant_color_bgr
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Water Region with Dominant Color", use_container_width=True)

                show_color_advisory(dominant_color_rgb)
                break

            if not found:
                st.warning("No water region detected or not enough dark pixels.")


