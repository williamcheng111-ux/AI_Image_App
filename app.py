import os
import cv2
import numpy as np
import streamlit as st

# ---------------------------------------------------------
# 1. 全局配置與 ONNX 模型載入
# ---------------------------------------------------------
st.set_page_config(page_title="AI 影像處理專題範例 App", layout="wide")

YUNET_MODEL_FILE = "face_detection_yunet_2023mar.onnx"

@st.cache_resource
def load_yunet_model():
    """安全載入 OpenCV ONNX YuNet 模型並進行快取"""
    if not os.path.exists(YUNET_MODEL_FILE):
        return None
    try:
        return cv2.FaceDetectorYN.create(
            model=YUNET_MODEL_FILE,
            config="",
            input_size=(320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000
        )
    except Exception:
        return None

face_detector = load_yunet_model()

# ---------------------------------------------------------
# 2. 側邊欄：檔案上傳
# ---------------------------------------------------------
st.sidebar.header("影像載入與解析")
uploaded_file = st.sidebar.file_uploader("選擇測試圖片", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is None:
    st.info("👈 請從左側選單上傳 PNG、JPG 或 BMP 圖片以開始進行分析。")
else:
    # 讀取並轉換圖片為 RGB NumPy 陣列
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ---------------------------------------------------------
    # 3. 主要 Tab 分頁
    # ---------------------------------------------------------
    basic_tab, color_tab, filter_tab, face_tab = st.tabs([
        "基礎資訊", "色彩空間與動態直方圖", "自訂卷積與 PSNR", "人臉偵測 (ONNX)"
    ])

    # Tab 1: 基礎資訊
    with basic_tab:
        h, w, c = img_rgb.shape
        col1, col2, col3 = st.columns(3)
        col1.metric("圖片寬度 (Width)", f"{w} px")
        col2.metric("圖片高度 (Height)", f"{h} px")
        col3.metric("通道數量 (Channels)", f"{c}")
        st.image(img_rgb, caption="原始上傳影像", width="stretch")

    # Tab 2: 色彩空間與直方圖
    with color_tab:
        st.subheader("色彩空間轉換")
        gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        st.image(gray_img, caption="灰階影像 (Grayscale)", width="stretch")

    # Tab 3: 自訂卷積與 PSNR
    with filter_tab:
        st.subheader("影像模糊與 PSNR 評估")
        blurred_img = cv2.GaussianBlur(img_rgb, (15, 15), 0)
        
        # 計算 MSE 與 PSNR
        mse = np.mean((img_rgb.astype(float) - blurred_img.astype(float)) ** 2)
        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100
        
        st.metric("PSNR 數值", f"{psnr:.2f} dB")
        st.image(blurred_img, caption="高斯模糊處理結果", width="stretch")

    # Tab 4: ONNX 人臉偵測
    with face_tab:
        st.subheader("ONNX YuNet 深度學習人臉偵測")
        if face_detector is None:
            st.error("⚠️ 找不到 ONNX 模型檔，請確認已上傳 `face_detection_yunet_2023mar.onnx`。")
        else:
            # 動態設定尺寸並執行偵測
            h, w, _ = img_rgb.shape
            face_detector.setInputSize((w, h))
            _, faces = face_detector.detect(img_rgb)
            
            annotated_img = img_rgb.copy()
            face_count = 0 if faces is None else len(faces)
            
            if faces is not None:
                for face in faces:
                    box = list(map(int, face[:4]))
                    score = face[14]
                    # 繪製邊界框與 Confidence
                    cv2.rectangle(annotated_img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 3)
                    cv2.putText(annotated_img, f"{score:.2f}", (box[0], max(0, box[1] - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            st.metric("偵測到的總人臉數", f"{face_count} 人")
            st.image(annotated_img, caption="人臉標註結果", width="stretch")