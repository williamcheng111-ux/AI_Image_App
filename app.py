"""
基於 AI 輔助之互動式影像處理與人臉分析工具
"""
from __future__ import annotations

from io import BytesIO
import math
import os
import urllib.request
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError
import streamlit as st

# 限定可上傳的副檔名與格式
ALLOWED_EXTENSIONS = ("png", "jpg", "jpeg", "bmp")
ALLOWED_IMAGE_FORMATS = {"PNG", "JPEG", "BMP"}


def load_image(file_bytes: bytes) -> tuple[Image.Image, int, np.ndarray]:
    """驗證上傳內容並回傳 PIL 圖片、通道數與 NumPy RGB 陣列。"""
    if not file_bytes:
        raise ValueError("檔案內容是空的，請重新選擇圖片。")

    with Image.open(BytesIO(file_bytes)) as source_image:
        if source_image.format not in ALLOWED_IMAGE_FORMATS:
            raise ValueError("不支援此圖片格式，請上傳 PNG、JPG 或 BMP。")

        source_image.load()
        width, height = source_image.size
        if width <= 0 or height <= 0:
            raise ValueError("圖片解析度無效，請改用其他圖片。")

        if source_image.mode == "P":
            channels = 4 if "transparency" in source_image.info else 3
        else:
            channels = len(source_image.getbands())

        # 轉為標準 RGB NumPy 陣列供 OpenCV 處理
        rgb_image = source_image.convert("RGB")
        img_np = np.array(rgb_image)

        return source_image.copy(), channels, img_np


@st.cache_resource
def get_yunet_onnx_model_path() -> str:
    """優先讀取本機 ONNX 模型檔，若不存在則自動下載並快取。"""
    onnx_path = "face_detection_yunet_2023mar.onnx"
    onnx_url = "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

    if not os.path.exists(onnx_path) or os.path.getsize(onnx_path) < 10000:
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(onnx_url, headers=headers)
        with urllib.request.urlopen(req) as resp, open(onnx_path, 'wb') as f:
            f.write(resp.read())

    return os.path.abspath(onnx_path)


# 頁面設定
st.set_page_config(page_title="AI 互動式影像處理與人臉分析工具", page_icon="📷", layout="wide")

# 側邊欄（已移除 "1."）
with st.sidebar:
    st.header("影像載入與解析")
    uploaded_file = st.file_uploader(
        "選擇圖片",
        type=list(ALLOWED_EXTENSIONS),
        help="支援 PNG、JPG、JPEG 與 BMP 格式。",
    )

st.title("📷 基於 AI 輔助之互動式影像處理與人臉分析工具")

# 預先建立四大分頁
basic_tab, color_tab, convolution_tab, face_tab = st.tabs([
    "基礎資訊", "1. 色彩空間與動態直方圖", "2. 自訂卷積與 PSNR", "3. AI 人臉偵測"
])

if uploaded_file is None:
    st.info("👈 請從左側選單上傳 PNG、JPG 或 BMP 圖片以開始進行分析。", icon="ℹ️")
else:
    image_bytes = uploaded_file.getvalue()
    try:
        image, channel_count, img_np = load_image(image_bytes)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        st.error(f"無法載入圖片：{error}")
    else:
        width, height = image.size
        file_size_kb = len(image_bytes) / 1024

        # ---------------------------------------------------------
        # Tab 0: 基礎資訊
        # ---------------------------------------------------------
        with basic_tab:
            st.subheader("圖片預覽與中繼資訊")
            st.image(image, caption="原始影像預覽", width="stretch")
            res_col, chan_col, size_col = st.columns(3)
            res_col.metric("解析度（Width × Height）", f"{width} × {height}")
            chan_col.metric("色彩通道數（Channels）", str(channel_count))
            size_col.metric("檔案大小", f"{file_size_kb:.2f} KB")
            st.caption(f"檔名：{uploaded_file.name}")

        # ---------------------------------------------------------
        # Tab 1: 色彩空間與動態直方圖分析
        # ---------------------------------------------------------
        with color_tab:
            st.subheader("多色彩空間切換與動態直方圖分析")
            mode = st.radio(
                "選擇檢視模式",
                ["Original (RGB)", "RGB 獨立通道", "YUV 色彩空間"],
                horizontal=True
            )

            if mode == "Original (RGB)":
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                ax[0].imshow(img_np)
                ax[0].set_title("Original RGB Image")
                ax[0].axis("off")

                colors = ('r', 'g', 'b')
                for i, col in enumerate(colors):
                    hist = cv2.calcHist([img_np], [i], None, [256], [0, 256])
                    ax[1].plot(hist, color=col, label=f'Channel {col.upper()}')
                ax[1].set_xlim([0, 256])
                ax[1].set_title("RGB Histogram")
                ax[1].set_xlabel("Pixel Intensity (0-255)")
                ax[1].set_ylabel("Pixel Count")
                ax[1].legend()
                st.pyplot(fig)

            elif mode == "RGB 獨立通道":
                R, G, B = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
                fig, axes = plt.subplots(2, 3, figsize=(15, 7))

                channels_data = [("Red Channel", R, 'red'), ("Green Channel", G, 'green'), ("Blue Channel", B, 'blue')]
                for idx, (title, ch_img, color_code) in enumerate(channels_data):
                    axes[0, idx].imshow(ch_img, cmap='gray')
                    axes[0, idx].set_title(title)
                    axes[0, idx].axis("off")

                    hist = cv2.calcHist([ch_img], [0], None, [256], [0, 256])
                    axes[1, idx].plot(hist, color=color_code)
                    axes[1, idx].set_xlim([0, 256])
                    axes[1, idx].set_title(f"{title} Histogram")
                    axes[1, idx].set_xlabel("Pixel Intensity")
                    axes[1, idx].set_ylabel("Pixel Count")

                plt.tight_layout()
                st.pyplot(fig)

            elif mode == "YUV 色彩空間":
                img_yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
                Y, U, V = img_yuv[:, :, 0], img_yuv[:, :, 1], img_yuv[:, :, 2]

                fig, axes = plt.subplots(2, 3, figsize=(15, 7))
                yuv_data = [("Y Component (Luminance)", Y), ("U Component (Chrominance)", U), ("V Component (Chrominance)", V)]

                for idx, (title, ch_img) in enumerate(yuv_data):
                    axes[0, idx].imshow(ch_img, cmap='gray')
                    axes[0, idx].set_title(title)
                    axes[0, idx].axis("off")

                    hist = cv2.calcHist([ch_img], [0], None, [256], [0, 256])
                    axes[1, idx].plot(hist, color='black')
                    axes[1, idx].set_xlim([0, 256])
                    axes[1, idx].set_title(f"{title} Histogram")
                    axes[1, idx].set_xlabel("Pixel Intensity")
                    axes[1, idx].set_ylabel("Pixel Count")

                plt.tight_layout()
                st.pyplot(fig)

        # ---------------------------------------------------------
        # Tab 2: 自訂卷積與 PSNR 計算
        # ---------------------------------------------------------
        with convolution_tab:
            st.subheader("自訂卷積核 (Kernel Matrix) 空間濾波與 PSNR 分析")

            col_preset, col_norm = st.columns([2, 1])
            with col_preset:
                preset = st.selectbox(
                    "快速選擇預設濾波核",
                    ["自訂 (Custom)", "邊緣銳化 (Sharpen)", "高斯模糊 (Gaussian Blur 3x3)", "浮雕效果 (Emboss)"]
                )
            with col_norm:
                do_normalize = st.checkbox("自動正規化 (Normalization)", value=True, help="將卷積核數值總和正規化為 1，避免亮度過曝")

            default_matrix_str = "0, -1, 0\n-1, 5, -1\n0, -1, 0"
            if preset == "邊緣銳化 (Sharpen)":
                default_matrix_str = "0, -1, 0\n-1, 5, -1\n0, -1, 0"
            elif preset == "高斯模糊 (Gaussian Blur 3x3)":
                default_matrix_str = "1, 2, 1\n2, 4, 2\n1, 2, 1"
            elif preset == "浮雕效果 (Emboss)":
                default_matrix_str = "-2, -1, 0\n-1, 1, 1\n0, 1, 2"

            kernel_input = st.text_area("輸入 N x N 矩陣（逗號/空格分隔元素，換行代表列）", value=default_matrix_str, height=120)

            try:
                lines = kernel_input.strip().split('\n')
                kernel_list = []
                for line in lines:
                    if line.strip():
                        row = [float(x) for x in line.replace(',', ' ').split()]
                        kernel_list.append(row)

                kernel = np.array(kernel_list, dtype=np.float32)

                # 防呆：檢查是否為二維方陣
                if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
                    st.error("⚠️ 防呆警告：卷積核必須為 N x N 的二維正方形矩陣！")
                else:
                    kernel_sum = np.sum(kernel)
                    if do_normalize:
                        if kernel_sum != 0:
                            kernel = kernel / kernel_sum
                        else:
                            st.warning("⚠️ 防呆警告：卷積核權重總和為 0，跳過正規化以防除以零錯誤。")

                    filtered_img = cv2.filter2D(img_np, -1, kernel)
                    filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)

                    # 算 MSE 與 PSNR
                    mse = np.mean((img_np.astype(np.float64) - filtered_img.astype(np.float64)) ** 2)
                    if mse == 0:
                        psnr = float('inf')
                    else:
                        psnr = 20 * math.log10(255.0 / math.sqrt(mse))

                    col_i1, col_i2 = st.columns(2)
                    with col_i1:
                        st.image(img_np, caption="原始影像", width="stretch")
                    with col_i2:
                        st.image(filtered_img, caption="濾波後結果", width="stretch")

                    # --- 放大的 PSNR 顯示卡片 ---
                    st.markdown("---")
                    if psnr == float('inf'):
                        st.metric("📊 峰值信噪比 (PSNR)", "∞ dB", delta="完全無失真")
                    else:
                        st.metric("📊 峰值信噪比 (PSNR)", f"{psnr:.2f} dB")

                    with st.expander("📐 查看 PSNR 計算公式與說明"):
                        st.latex(r"MSE = \frac{1}{3 m n} \sum_{c=1}^{3} \sum_{i=0}^{m-1} \sum_{j=0}^{n-1} [I(i,j,c) - K(i,j,c)]^2")
                        st.latex(r"PSNR = 10 \cdot \log_{10} \left( \frac{255^2}{MSE} \right) = 20 \cdot \log_{10} \left( \frac{255}{\sqrt{MSE}} \right)")

            except Exception as e:
                st.error(f"⚠️ 解析矩陣失敗：請確保格式正確，輸入皆為數字。（訊息: {e}）")

        # ---------------------------------------------------------
        # Tab 3: 人臉偵測 (OpenCV ONNX YuNet 深度學習模型)
        # ---------------------------------------------------------
        with face_tab:
            st.subheader("OpenCV ONNX YuNet AI 人臉偵測模組")

            conf_threshold = st.slider("調整偵測置信度門檻 (Confidence Threshold)", 0.1, 1.0, 0.5, 0.05)

            annotated_img = img_np.copy()
            # OpenCV FaceDetectorYN 接收 BGR 格式
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            face_count = 0
            face_details = []

            try:
                model_path = get_yunet_onnx_model_path()
                detector = cv2.FaceDetectorYN.create(
                    model=model_path,
                    config="",
                    input_size=(width, height),
                    score_threshold=float(conf_threshold),
                    nms_threshold=0.3,
                    top_k=500
                )

                _, faces = detector.detect(img_bgr)

                if faces is not None:
                    face_count = len(faces)
                    for idx, face in enumerate(faces):
                        # 人臉座標 (x, y, w, h)
                        box = list(map(int, face[0:4]))
                        confidence = float(face[-1])
                        
                        xmin, ymin, bw, bh = box
                        xmin, ymin = max(0, xmin), max(0, ymin)

                        # 畫綠色人臉框與標籤
                        cv2.rectangle(annotated_img, (xmin, ymin), (xmin + bw, ymin + bh), (0, 255, 0), 3)
                        label = f"Face #{idx+1}: {confidence*100:.1f}%"
                        cv2.putText(annotated_img, label, (xmin, max(ymin - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        face_details.append({
                            "人臉編號": idx + 1,
                            "置信度 (Confidence)": f"{confidence*100:.2f}%",
                            "邊界框 (x, y, w, h)": f"({xmin}, {ymin}, {bw}, {bh})"
                        })

                st.caption("🤖 偵測技術：OpenCV YuNet Deep Learning Face Detector (ONNX Standard)")

            except Exception as e:
                st.error(f"⚠️ 載入或執行 ONNX 模型時發生錯誤：{e}")

            col_f1, col_f2 = st.columns([2, 1])
            with col_f1:
                st.image(annotated_img, caption="人臉偵測結果標註圖", width="stretch")
            with col_f2:
                st.metric("偵測到的總人臉數量", f"{face_count} 人")
                if face_count > 0:
                    st.dataframe(face_details)
                else:
                    st.warning("未偵測到人臉，可嘗試調低置信度門檻。")