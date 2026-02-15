import streamlit as st
import cv2
import numpy as np
import joblib
import mediapipe as mp
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import time
import os
from googletrans import Translator  # مكتبة الترجمة

# ==========================
# 1. إعداد الصفحة والواجهة
# ==========================
st.set_page_config(page_title="ASL AI Recognition", layout="wide", page_icon="🖐️")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .title-text {
        text-align: center;
        color: #00FFCC;
        font-size: 50px;
        font-weight: bold;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px #000000;
    }
    .warning-box {
        background-color: #ff4b4b;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 25px;
        border: 2px solid white;
    }
    .output-box {
        padding: 20px;
        background-color: #1f2937;
        border-radius: 10px;
        border: 2px solid #00FFCC;
        color: #ffffff;
        font-size: 28px;
        text-align: center;
        min-height: 100px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="title-text">ASL Hand Sign Interpreter</p>', unsafe_allow_html=True)
st.markdown('<div class="warning-box">⚠️ يرجى استخدام "اليد اليسرى" فقط لضمان دقة التعرف</div>', unsafe_allow_html=True)

# ==========================
# 2. مسارات الموديلات
# ==========================
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "model", "asl_hand_model.keras")
scaler_path = os.path.join(BASE_DIR, "model", "scaler.save")
le_path = os.path.join(BASE_DIR, "model", "label_encoder.save")
hand_task_path = os.path.join(BASE_DIR, "model", "hand_landmarker.task")

@st.cache_resource
def load_all_models():
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    le = joblib.load(le_path)
    return model, scaler, le

model, scaler, le = load_all_models()

# ==========================
# 3. إعداد MediaPipe
# ==========================
mp_tasks = mp.tasks
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp_tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=hand_task_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

# ==========================
# 4. Hand connections
# ==========================
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

# ==========================
# 5. Video Processor
# ==========================
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.landmarker = HandLandmarker.create_from_options(options)
        self.recorded_letters = []
        self.last_hand_points = None
        self.stable_start_time = None
        self.last_added_time = 0
        self.STABLE_THRESHOLD = 0.04
        self.NORMAL_DURATION = 1.5  # ثبات اليد قبل تسجيل الحرف
        self.REPEAT_DURATION = 2.0  # تكرار الحرف بعد 2 ثانية إذا استمر الثبات

    def is_hand_stable(self, current, last):
        if last is None:
            return False
        return np.mean(np.abs(np.array(current) - np.array(last))) < self.STABLE_THRESHOLD

    def draw_ui(self, frame, current_letter, full_text):
        h, w, _ = frame.shape
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-70), (w, h), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, f"Text: {full_text}", (20, h-25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 204), 2)
        
        cv2.rectangle(frame, (w-120, 20), (w-20, 120), (0, 255, 204), 2)
        cv2.rectangle(frame, (w-118, 22), (w-22, 118), (0,0,0), -1)
        if current_letter:
            cv2.putText(frame, current_letter, (w-105, 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 204), 4)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        pred_class = ""
        feature_vector = []

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            for pt in hand:
                feature_vector.extend([pt.x, pt.y, pt.z])

            landmark_points = [(int(pt.x*w), int(pt.y*h)) for pt in hand]
            for s, e in HAND_CONNECTIONS:
                cv2.line(img, landmark_points[s], landmark_points[e], (0,255,0), 2)
            for i, (px, py) in enumerate(landmark_points):
                color = (0,0,255) if i in [4,8,12,16,20] else (255,0,0)
                cv2.circle(img, (px, py), 5, color, -1)

            fv = np.array(feature_vector).reshape(1, -1)
            fv = scaler.transform(fv)
            pred_index = np.argmax(model.predict(fv, verbose=0))
            pred_class = le.inverse_transform([pred_index])[0]

            now = time.time()
            if self.is_hand_stable(feature_vector, self.last_hand_points):
                if self.stable_start_time is None:
                    self.stable_start_time = now
                elapsed = now - self.stable_start_time
                if elapsed >= self.NORMAL_DURATION:
                    if not self.recorded_letters or pred_class != self.recorded_letters[-1]:
                        self.recorded_letters.append(pred_class)
                        self.last_added_time = now
                    elif now - self.last_added_time >= self.REPEAT_DURATION:
                        self.recorded_letters.append(pred_class)
                        self.last_added_time = now
            else:
                self.stable_start_time = None

            self.last_hand_points = feature_vector.copy()

        self.draw_ui(img, pred_class, "".join(self.recorded_letters))
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================
# 6. واجهة المستخدم
# ==========================
translator = Translator()  # مترجم جوجل

c1, c2 = st.columns([2,1])

with c1:
    st.subheader("🎥 الكاميرا المباشرة")
    ctx = webrtc_streamer(
        key="asl-main",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

with c2:
    st.subheader("📋 النص المستخرج")
    text_area_original = st.empty()
    text_area_translated = st.empty()  # صندوق جديد للترجمة

    if st.button("🔄 تحديث النص"):
        if ctx.video_processor:
            txt = "".join(ctx.video_processor.recorded_letters)
            text_area_original.markdown(f'<div class="output-box">{txt}</div>', unsafe_allow_html=True)

    if st.button("🗑️ مسح الكل"):
        if ctx.video_processor:
            ctx.video_processor.recorded_letters = []
            text_area_original.markdown('<div class="output-box"></div>', unsafe_allow_html=True)
            text_area_translated.markdown('<div class="output-box"></div>', unsafe_allow_html=True)

    if st.button("🌐 ترجمة إلى العربية"):
        if ctx.video_processor:
            original_text = "".join(ctx.video_processor.recorded_letters)
            if original_text:
                translated = translator.translate(original_text, src='en', dest='ar')
                text_area_translated.markdown(f'<div class="output-box">{translated.text}</div>', unsafe_allow_html=True)
            else:
                st.warning("لا يوجد نص لترجمته")

    st.info("💡 النظام يسجل حرفاً جديداً بعد 1.5 ثانية من ثبات اليد ويكرر نفس الحرف كل 2 ثانية إذا بقيت اليد ثابتة.")
