import os
import cv2
import streamlit as st

YUNET_MODEL_FILE = "face_detection_yunet_2023mar.onnx"

@st.cache_resource
def load_yunet_model():
    if not os.path.exists(YUNET_MODEL_FILE):
        return None
    try:
        # 只保留最核心的標準參數，絕對不要設定 setPreferableTarget/setPreferableBackend
        detector = cv2.FaceDetectorYN.create(
            model=YUNET_MODEL_FILE,
            config="",
            input_size=(300, 300),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000
        )
        return detector
    except Exception as e:
        st.error(f"模型初始化失敗: {e}")
        return None

face_detector = load_yunet_model()