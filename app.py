import os
import cv2
import av
import numpy as np
import pickle
import streamlit as st
import mediapipe as mp

# Try importing TensorFlow; handle failure gracefully if not installed correctly
try:
    import tensorflow as tf
except ImportError:
    st.error("TensorFlow is not installed. Please check requirements.txt.")
    st.stop()

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ==========================================
# CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AI Sign Language Translator",
    page_icon="🤟",
    layout="centered"
)

# Force CPU usage to prevent GPU memory errors on Cloud instances
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Constants
STABILITY_FRAMES = 15
CONFIDENCE_THRESHOLD = 0.85
MODEL_PATH = "sign_language_model.keras"
ENCODER_PATH = "label_encoder.pkl"

# STUN Server Configuration (CRITICAL for Deployment)
# This allows the video stream to navigate through firewalls/NATs.
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ==========================================
# RESOURCE LOADING (CACHED)
# ==========================================
@st.cache_resource
def load_resources():
    """
    Load Model, Encoder, and MediaPipe safely.
    Returns: (model, encoder, hands_solution, draw_utils, hands_style) or None
    """
    # 1. Check File Existence
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        return None

    try:
        # 2. Load Keras Model
        model = tf.keras.models.load_model(MODEL_PATH)
        
        # 3. Load Label Encoder
        with open(ENCODER_PATH, "rb") as f:
            encoder = pickle.load(f)

        # 4. Setup MediaPipe
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        return model, encoder, hands, mp_hands, mp_drawing
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None

# Load Resources
resources = load_resources()

# ==========================================
# UI & LOGIC
# ==========================================
st.title("🤟 AI Sign Language Translator")

if resources is None:
    st.error("❌ Model files not found!")
    st.warning(f"""
    **Deployment Check:**
    Ensure these files are in the root directory (same folder as app.py):
    1. `{MODEL_PATH}`
    2. `{ENCODER_PATH}`
    """)
    st.stop()
else:
    model, encoder, hands, mp_hands, mp_drawing = resources

class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.sentence = []
        self.last_pred = None
        self.frame_counter = 0

    def recv(self, frame):
        # 1. Convert Frame
        img = frame.to_ndarray(format="bgr24")
        
        # 2. Pre-processing
        image = cv2.flip(img, 1) # Mirror effect
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 3. Hand Detection
        results = hands.process(rgb)
        
        predicted_char = "..."
        hand_detected = False

        if results.multi_hand_landmarks:
            hand_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                # Visuals: Draw Landmarks
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )

                # Extract Data
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])

                # AI Prediction
                try:
                    input_data = np.array([landmarks], dtype=np.float32)
                    prediction = model.predict(input_data, verbose=0)
                    class_id = np.argmax(prediction)
                    confidence = np.max(prediction)

                    if confidence > CONFIDENCE_THRESHOLD:
                        current_char = encoder.inverse_transform([class_id])[0]
                        
                        # Stabilization Logic
                        if current_char == self.last_pred:
                            self.frame_counter += 1
                        else:
                            self.frame_counter = 0
                            self.last_pred = current_char

                        if self.frame_counter >= STABILITY_FRAMES:
                            p = current_char.lower()
                            if "space" in p:
                                self.sentence.append(" ")
                            elif "delete" in p:
                                if self.sentence: self.sentence.pop()
                            else:
                                self.sentence.append(current_char)
                            
                            self.frame_counter = 0
                            predicted_char = current_char # Update immediately
                        else:
                            predicted_char = self.last_pred if self.last_pred else "..."
                except Exception:
                    pass # Prevent crashing on prediction error

        # ==========================================
        # OVERLAY VISUALS (High Visibility)
        # ==========================================
        h, w, _ = image.shape
        
        # Top Bar Background
        cv2.rectangle(image, (0, 0), (w, 100), (20, 20, 20), -1)
        
        # Status Light
        status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
        cv2.circle(image, (w - 30, 30), 10, status_color, -1)

        # Text: Prediction
        cv2.putText(image, "DETECTING:", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(image, f"{predicted_char}", (150, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

        # Text: Sentence
        sentence_str = "".join(self.sentence)[-20:] # Show last 20 chars
        cv2.putText(image, "SENTENCE:", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(image, sentence_str, (150, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Return to Streamlit
        return av.VideoFrame.from_ndarray(image, format="bgr24")

st.markdown("### 📷 Camera Feed")
st.info("Ensure you are in a well-lit environment and your hand is clearly visible.")

webrtc_streamer(
    key="sign-translator",
    mode=webrtc_streamer.WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=SignLanguageProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

st.markdown("---")
st.markdown("Made with Streamlit, TensorFlow & MediaPipe")