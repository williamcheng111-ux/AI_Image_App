import os
import cv2
import streamlit as st

YUNET_MODEL_FILE = "face_detection_yunet_2023mar.onnx"

@st.cache_resource
def load_yunet_model():
    if not os.path.exists(YUNET_MODEL_FILE):
        st.error(f"找不到模型檔案：{YUNET_MODEL_FILE}，請確認已推送到 GitHub！")
        return None
    
    try:
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
        st.error(f"載入 YuNet 模型失敗: {e}")
        return None

# 調用模型
face_detector = load_yunet_model()