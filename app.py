import os
import urllib.request
import cv2
import streamlit as st

# 定義 ONNX 模型檔名與官方下載 URL
YUNET_MODEL_FILE = "face_detection_yunet_2023mar.onnx"
YUNET_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

# 防呆機制：若本機或雲端沒有該模型檔，自動下發 HTTP 請求下載
if not os.path.exists(YUNET_MODEL_FILE):
    with st.spinner("首次執行，正在自動下載 OpenCV ONNX YuNet 模型檔..."):
        urllib.request.urlretrieve(YUNET_MODEL_URL, YUNET_MODEL_FILE)

# 順利載入模型（確保不會再報 Can't read ONNX file 錯誤）
face_detector = cv2.FaceDetectorYN.create(
    model=YUNET_MODEL_FILE,
    config="",
    input_size=(300, 300),
    score_threshold=0.6,
    nms_threshold=0.3,
    top_k=5000
)