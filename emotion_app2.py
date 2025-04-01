import streamlit as st
st.set_page_config(page_title="🎭 Emotion Recognition System", layout="centered")

import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import gdown
from torchvision import transforms, models
from PIL import Image
from collections import deque, Counter
import plotly.express as px
import pandas as pd
import time

# ==== Constants ====
EMOTIONS = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
POSITIVE_EMOTIONS = ['Happy', 'Surprise', 'Neutral']
NEGATIVE_EMOTIONS = ['Angry', 'Sad', 'Fear']

# ==== Load model from Google Drive ====
@st.cache_resource
def load_model():
    MODEL_PATH = "best_emotion_model_gray48_1.pth"
    FILE_ID = "1h6OZWxlWDr_IDzlb4LucQzWV56kSqgza"
    DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"

    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading model from Google Drive..."):
            gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)

    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 6)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ==== Traffic light logic ====
def get_quality_status(recent_emotions):
    last = list(recent_emotions)[-10:]
    neg = sum(1 for e in last if e in NEGATIVE_EMOTIONS)
    if neg >= 7:
        return "🔴 Dissatisfied", (0, 0, 255)
    elif neg >= 4:
        return "🟡 Doubtful", (0, 255, 255)
    else:
        return "🟢 Satisfied", (0, 255, 0)

# ==== Streamlit UI ====
st.title("🎥 Real-Time Emotion Recognition")
FRAME_WINDOW = st.image([])
emotion_display = st.empty()
traffic_light_display = st.empty()
timer_display = st.empty()

# ==== State ====
if "running" not in st.session_state:
    st.session_state.running = False
    st.session_state.start_time = None
    st.session_state.emotion_log = []
    st.session_state.recent_emotions = deque(maxlen=20)

start_button = st.button("▶️ Start Session")
stop_button = st.button("⏹️ Stop Session")

# ==== Start camera session ====
if start_button:
    st.session_state.running = True
    st.session_state.start_time = time.time()
    st.session_state.emotion_log = []
    st.session_state.recent_emotions = deque(maxlen=20)

    cap = cv2.VideoCapture(0)

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.warning("❌ Camera not available.")
            break

        elapsed = int(time.time() - st.session_state.start_time)
        timer_display.markdown(f"⏱️ Session Duration: `{elapsed}` seconds")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_pil = Image.fromarray(face_img).convert("L")
            input_tensor = transform(face_pil).unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
                confidence, pred = torch.max(probs, 1)
                emotion = EMOTIONS[pred.item()]
                confidence_percent = int(confidence.item() * 100)
                label_text = f"{emotion} ({confidence_percent}%)"

            st.session_state.emotion_log.append({
                "timestamp": elapsed,
                "emotion": emotion
            })
            st.session_state.recent_emotions.append(emotion)

            color = (0, 255, 0) if emotion in POSITIVE_EMOTIONS else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if st.session_state.recent_emotions:
            verdict, box_color = get_quality_status(st.session_state.recent_emotions)
            cv2.rectangle(frame, (10, 10), (260, 60), box_color, -1)
            cv2.putText(frame, verdict, (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        emotion_display.markdown(f"😊 **Detected Emotion:** `{label_text}`")
        traffic_light_display.markdown(f"🚦 **Service Quality (Live):** `{verdict}`")

        if st.session_state.running is False:
            break

        time.sleep(0.05)

    cap.release()

if stop_button:
    st.session_state.running = False

# ==== Results after session ====
if not st.session_state.running and st.session_state.emotion_log:
    st.subheader("📋 Emotion Log")
    df = pd.DataFrame(st.session_state.emotion_log)
    st.dataframe(df)

    st.subheader("📈 Emotion Timeline")
    emotion_to_id = {e: i for i, e in enumerate(EMOTIONS)}
    df["emotion_id"] = df["emotion"].map(emotion_to_id)

    fig = px.line(df, x="timestamp", y="emotion_id", markers=True,
                  labels={"timestamp": "Time (sec)", "emotion_id": "Emotion"},
                  title="Client Emotion Over Time")
    fig.update_yaxes(
        tickvals=list(emotion_to_id.values()),
        ticktext=list(emotion_to_id.keys())
    )
    st.plotly_chart(fig)

    df.to_csv("emotion_logs.csv", index=False)

    st.subheader("📌 Final Verdict: Was the Client Satisfied?")
    counts = df["emotion"].value_counts()
    total = counts.sum()
    negative = sum(counts.get(e, 0) for e in NEGATIVE_EMOTIONS)
    positive = sum(counts.get(e, 0) for e in POSITIVE_EMOTIONS)

    if total == 0:
        final_result = "🟡 Unable to determine (not enough data)"
    elif negative / total >= 0.5:
        final_result = "🔴 Client is **NOT satisfied**"
    elif positive / total >= 0.6:
        final_result = "🟢 Client is **satisfied**"
    else:
        final_result = "🟡 Client status is **uncertain**"

    st.markdown(f"### {final_result}")
