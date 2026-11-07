import os
import cv2
import numpy as np
import streamlit as st

# 1. 宣告 ONNX 檔案路徑
YUNET_MODEL_FILE = "face_detection_yunet_2023mar.onnx"

# 2. 安全載入模型（加入 try-except，避免任何崩潰造成網頁全白）
@st.cache_resource
def load_yunet_model():
    if not os.path.exists(YUNET_MODEL_FILE):
        return None
    try:
        # 使用標準預設參數
        detector = cv2.FaceDetectorYN.create(
            model=YUNET_MODEL_FILE,
            config="",
            input_size=(320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000
        )
        return detector
    except Exception as e:
        return None

# 初始化模型
face_detector = load_yunet_model()

# --- Tab 3 人臉偵測執行邏輯 ---
# （請確認在 face_tab 內執行）
if face_detector is None:
    st.warning("⚠️ 尚未偵測到 ONNX 模型檔或模型載入中，請確認 `face_detection_yunet_2023mar.onnx` 已推送到 GitHub。")
else:
    # 取得上傳的圖片尺寸並動態調整 YuNet 的 Input Size
    h, w, _ = img_np.shape
    face_detector.setInputSize((w, h))
    
    # 執行偵測
    _, faces = face_detector.detect(img_np)
    
    # 後續繪製與 st.metric 呈現...