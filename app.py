import streamlit as st
import os
import tempfile
import numpy as np
import gdown
from PIL import Image
from ultralytics import YOLO

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH = "best.pt"
GDRIVE_FILE_ID = "1rJ7xvaDRg_vpFr7ZGt76qRrUhgjGzU_8"

DISEASE_INFO = {
    "Apple Scab": "Fungal disease causing dark, scabby lesions on leaves and fruit.",
    "Apple Rust": "Rust fungus that creates orange/yellow spots on apple leaves.",
    "Corn Leaf Blight": "Fungal disease causing long, tan lesions on corn leaves.",
    "Corn Gray Spot": "Causes rectangular gray/brown lesions running parallel to leaf veins.",
    "Potato Early Blight": "Dark brown spots with concentric rings, usually on older leaves.",
    "Potato Late Blight": "Water-soaked lesions that turn brown/black — highly destructive.",
    "Tomato Bacterial Spot": "Small, water-soaked spots that turn brown with yellow halos.",
    "Tomato Leaf Mold": "Pale green/yellow patches on upper leaf surface, mold on underside.",
    "Tomato Mosaic Virus": "Mottled light/dark green pattern on leaves, stunted growth.",
    "Healthy Leaf": "No disease detected. The plant appears healthy!",
}

st.set_page_config(page_title="Plant Disease AI | Anushka Sahu", page_icon="🌿", layout="wide")

# ─────────────────────────────────────────────
# CUSTOM CSS (Cleaned HTML injection)
# ─────────────────────────────────────────────
custom_css = """
<style>
.stApp {
    background-color: #143601;
    color: #e8f5e9;
}
[data-testid="stSidebar"] {
    background-color: #1a4301;
    border-right: 2px solid #245501;
}
h1, h2, h3 {
    color: #aad576 !important;
}
.stButton>button {
    background-color: #538d22;
    color: white;
    border: none;
    border-radius: 6px;
    font-weight: bold;
}
.stButton>button:hover {
    background-color: #73a942;
    color: white;
    border: none;
}
div[data-testid="stExpander"] {
    background-color: #245501;
    border: 1px solid #538d22;
    border-radius: 6px;
}
div[data-testid="stExpander"] summary {
    color: #aad576;
    font-weight: bold;
}
div.stAlert {
    background-color: #245501;
    color: white;
    border: 1px solid #73a942;
}
.student-card {
    background-color: #245501;
    padding: 14px;
    border-radius: 8px;
    border-left: 5px solid #aad576;
    margin-bottom: 15px;
    color: #ffffff;
}
</style>
"""
st.html(custom_css)

# ─────────────────────────────────────────────
# LOAD MODEL (Auto-downloads if missing)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI model weights from Google Drive (First-time setup)..."):
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, output=MODEL_PATH, quiet=False)
    return YOLO(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def run_detection(model, image: Image.Image, conf_threshold: float):
    img_array = np.array(image)
    results = model.predict(img_array, conf=conf_threshold, verbose=False)
    result = results[0]

    annotated_bgr = result.plot()
    annotated_rgb = annotated_bgr[:, :, ::-1]

    detections = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        detections.append({"class": class_name, "confidence": confidence})

    return Image.fromarray(annotated_rgb), detections

def run_video_detection(model, video_path: str, conf_threshold: float):
    import imageio

    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data().get("fps", 25)

    all_detections = []
    frames = []
    MAX_FRAMES = 50

    progress = st.progress(0, text="Processing video frames...")

    for frame_count, frame in enumerate(reader):
        if frame_count >= MAX_FRAMES:
            break

        try:
            results = model.track(frame, persist=True, conf=conf_threshold, verbose=False)
        except Exception:
            results = model.predict(frame, conf=conf_threshold, verbose=False)

        annotated_bgr = results[0].plot()
        annotated_rgb = annotated_bgr[:, :, ::-1]

        frame_img = Image.fromarray(annotated_rgb).resize((640, 360))
        frames.append(frame_img)

        for box in results[0].boxes:
            if box.cls is not None:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                all_detections.append({"class": class_name, "confidence": confidence})

        progress.progress(
            (frame_count + 1) / MAX_FRAMES,
            text=f"Processing frame {frame_count + 1}/{MAX_FRAMES}..."
        )

    reader.close()
    progress.empty()

    out_path = video_path.replace(".mp4", "_detected.gif")
    if frames:
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            loop=0,
            duration=int(1000 / fps),
        )

    return out_path, all_detections

# ─────────────────────────────────────────────
# UI - SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.html("""
    <div class="student-card">
        <h3 style="margin: 0 0 6px 0; color: #aad576;">Science Exhibition</h3>
        <p style="margin: 2px 0;"><b>Name:</b> Anushka Sahu</p>
        <p style="margin: 2px 0;"><b>Class:</b> 11th D</p>
        <p style="margin: 2px 0;"><b>Roll:</b> 5</p>
    </div>
    """)

    st.markdown("### Settings")
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )
    
    st.divider()
    
    st.markdown("### Technology Stack")
    st.markdown("""
    - **Model:** Ultralytics YOLOv8
    - **Backend:** PyTorch
    - **Frontend:** Streamlit
    - **Processing:** OpenCV & Pillow
    - **Dataset:** Roboflow Universe
    """)

# ─────────────────────────────────────────────
# UI - MAIN PAGE
# ─────────────────────────────────────────────
st.title("🌿 Agricultural Disease AI")
st.markdown("Upload a leaf image or video to run inference using our custom-trained YOLOv8 neural network.")

tab_image, tab_video = st.tabs(["📷 Image Inference", "🎥 Video Inference"])

# ── IMAGE TAB ──
with tab_image:
    uploaded_image = st.file_uploader("Select Leaf Image", type=["jpg", "jpeg", "png"], key="img_upload")

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Input Sample")
            st.image(image, use_container_width=True)

        with st.spinner("Executing YOLOv8 inference..."):
            annotated_image, detections = run_detection(model, image, conf_threshold)

        with col2:
            st.markdown("### AI Output")
            st.image(annotated_image, use_container_width=True)

        if detections:
            st.markdown(f"### 🔍 Detected Conditions ({len(detections)})")
            for det in detections:
                conf_pct = det["confidence"] * 100
                disease = det["class"]
                info = DISEASE_INFO.get(disease, "Anomaly detected.")
                
                with st.expander(f"{disease} — {conf_pct:.1f}% Match"):
                    st.write(info)
        else:
            st.info("No supported diseases detected above the confidence threshold.")

# ── VIDEO TAB ──
with tab_video:
    uploaded_video = st.file_uploader("Select Field Video", type=["mp4", "mov", "avi"], key="vid_upload")

    if uploaded_video:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            tmp_path = tmp.name

        st.video(tmp_path)

        if st.button("Run Batch Inference", type="primary"):
            with st.spinner("Processing video frames..."):
                try:
                    out_path, detections = run_video_detection(model, tmp_path, conf_threshold)
                except Exception as e:
                    st.error(f"Inference failed: {e}")
                    os.unlink(tmp_path)
                    st.stop()

            st.success("Batch processing complete.")
            st.markdown("### Processed Feed")
            st.image(out_path, caption="Object Tracking Output")

            if detections:
                from collections import Counter
                counts = Counter(d["class"] for d in detections)
                avg_conf = {}
                for d in detections:
                    avg_conf.setdefault(d["class"], []).append(d["confidence"])

                st.markdown("### Aggregate Results")
                for cls, count in counts.most_common():
                    avg = sum(avg_conf[cls]) / len(avg_conf[cls]) * 100
                    st.markdown(f"- **{cls}**: Found in {count} frames (Average Confidence: {avg:.1f}%)")
            else:
                st.info("No anomalies detected across processed frames.")

            os.unlink(tmp_path)
