import json
import math
import cv2
from google import genai
from google.genai import types
import numpy as np
from PIL import Image
import streamlit as st

# ==========================================================
# 1. Streamlit Page Configuration
# ==========================================================
st.set_page_config(
    page_title="Gemini Hybrid Vision & Model Classifier",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 ระบบวัดมุม Auto-Crop และแยกโมเดลด้วย Gemini AI + OpenCV")
st.caption(
    "ผสานพลัง Multimodal AI และ Geometric Computer Vision เพื่อความแม่นยำสูงสุด"
)

# ==========================================================
# 2. Gemini API Initialization
# ==========================================================
st.sidebar.header("🔑 การตั้งค่าระบบ API")
api_key = st.sidebar.text_input(
    "Gemini API Key",
    value=st.secrets.get("GEMINI_API_KEY", ""),
    type="password",
    help="ระบุ API Key ของ Google Gemini",
)


# ==========================================================
# 3. Core Computer Vision Engine (OpenCV + Gemini)
# ==========================================================
def detect_precise_angle_opencv(image):
    """OpenCV Sub-pixel Angle & ROI Engine"""
    default_phi, default_roi = 0.0, None
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return default_phi, default_roi, None

    try:
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(image.shape) == 3
            else image.copy()
        )
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        v = np.median(blurred)
        edges = cv2.Canny(
            blurred, int(max(0, 0.67 * v)), int(min(255, 1.33 * v))
        )

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 360,
            threshold=30,
            minLineLength=20,
            maxLineGap=8,
        )

        if lines is None or len(lines) == 0:
            return default_phi, default_roi, edges

        angles, lengths, x_coords, y_coords = [], [], [], []
        for line in lines:
            if len(line) > 0 and len(line[0]) == 4:
                x1, y1, x2, y2 = line[0]
                length = math.hypot(x2 - x1, y2 - y1)
                if length < 15:
                    continue
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                angles.append(angle)
                lengths.append(length)
                x_coords.extend([x1, x2])
                y_coords.extend([y1, y2])

        if angles and x_coords and y_coords:
            weighted_angles = []
            for a, l in zip(angles, lengths):
                weighted_angles.extend([a] * int(l))

            cv_phi = float(np.median(weighted_angles))
            roi_box = (
                int(min(x_coords)),
                int(min(y_coords)),
                int(max(x_coords)),
                int(max(y_coords)),
            )
            return round(cv_phi, 2), roi_box, edges
        return default_phi, default_roi, edges
    except Exception:
        return default_phi, default_roi, None


def analyze_with_gemini(pil_image, key):
    """วิเคราะห์มุม พิกัด และจำแนกประเภทโมเดลขั้นสูงด้วย Gemini API"""
    if not key:
        return None

    try:
        client = genai.Client(api_key=key)

        prompt = """
        Analyze this image for high-precision model classification and angle measurement:
        1. Identify the main subject/model and classify its category accurately.
        2. Detect the exact orientation/tilt angle (in degrees from horizontal, -180 to 180).
        3. Identify the bounding box coordinates [ymin, xmin, ymax, xmax] normalized from 0 to 1000.
        4. List key structural visual characteristics observed.

        Respond STRICTLY in JSON format with this structure:
        {
            "model_type": "Classified Model Name / Category",
            "confidence_score": 95.5,
            "detected_angle": 12.4,
            "bounding_box_1000": [ymin, xmin, ymax, xmax],
            "structural_features": ["feature 1", "feature 2"],
            "analysis_reasoning": "Brief explanation of classification"
        }
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[pil_image, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,  # ค่าต่ำเพื่อความเที่ยงตรงและแม่นยำสูง
            ),
        )

        return json.loads(response.text)
    except Exception as e:
        st.error(f"Gemini API Error: {str(e)}")
        return None


def auto_crop_roi(image, roi_box, padding=15):
    """Crop ภาพอัตโนมัติ"""
    if image is None or roi_box is None:
        return None
    h, w = image.shape[:2]
    x1, y1, x2, y2 = roi_box
    return image[
        max(0, y1 - padding) : min(h, y2 + padding),
        max(0, x1 - padding) : min(w, x2 + padding),
    ]


# ==========================================================
# 4. Main Application Workflow
# ==========================================================
uploaded_file = st.file_uploader(
    "อัปโหลดไฟล์ภาพเพื่อเริ่มวิเคราะห์แบบ Hybrid AI (JPG, PNG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    # อ่านไฟล์ภาพสำหรับ OpenCV และ PIL
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()), dtype=np.uint8
    )
    cv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    pil_image = Image.open(uploaded_file)

    # 1. ประมวลผลจาก OpenCV Geometric Engine
    cv_phi, cv_roi, edges = detect_precise_angle_opencv(cv_image)

    # 2. ประมวลผลด้วย Gemini Multimodal AI Engine
    gemini_result = None
    if api_key:
        with st.spinner("🧠 กำลังให้ Gemini AI วิเคราะห์โมเดลและโครงสร้าง..."):
            gemini_result = analyze_with_gemini(pil_image, api_key)
    else:
        st.warning(
            "⚠️ กรุณาระบุ Gemini API Key ในแถบด้านซ้าย เพื่อเปิดใช้งานการวิเคราะห์ขั้นสูงด้วย AI"
        )

    # 3. รวมผลลัพธ์เพื่อความแม่นยำสูงสุด (Hybrid Fusion)
    final_roi = cv_roi
    final_angle = cv_phi
    model_name = "Unclassified (OpenCV Only)"
    confidence = 0.0
    reasoning = "ประมวลผลด้วย OpenCV แบบพื้นฐาน"
    features = []

    if gemini_result:
        model_name = gemini_result.get("model_type", "Unknown Model")
        confidence = gemini_result.get("confidence_score", 0.0)
        reasoning = gemini_result.get("analysis_reasoning", "")
        features = gemini_result.get("structural_features", [])

        # แปลง Bounding Box แบบ Normalized (0-1000) ของ Gemini มาเป็น Pixel Coordinates
        g_box = gemini_result.get("bounding_box_1000")
        if g_box and len(g_box) == 4:
            h, w = cv_image.shape[:2]
            g_ymin, g_xmin, g_ymax, g_xmax = g_box
            g_pixel_roi = (
                int((g_xmin / 1000.0) * w),
                int((g_ymin / 1000.0) * h),
                int((g_xmax / 1000.0) * w),
                int((g_ymax / 1000.0) * h),
            )
            final_roi = g_pixel_roi if cv_roi is None else cv_roi

        # ถ่วงน้ำหนักค่ามุมองศาร่วมกันระหว่าง OpenCV และ Gemini
        g_angle = gemini_result.get("detected_angle", cv_phi)
        final_angle = round((cv_phi * 0.6) + (g_angle * 0.4), 2)

    # Auto-Crop ภาพตามพิกัด ROI
    cropped_img = auto_crop_roi(cv_image, final_roi)

    # ------------------------------------------------------
    # 5. Dashboard Display
    # ------------------------------------------------------
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("1. ภาพต้นฉบับ & กรอบ ROI")
        annotated = cv_image.copy()
        if final_roi is not None:
            x1, y1, x2, y2 = final_roi
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
        st.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True
        )

    with col2:
        st.subheader("2. แผนผังขอบภาพ (Edges)")
        if edges is not None:
            st.image(edges, use_container_width=True)

    with col3:
        st.subheader("3. Auto-Crop ROI")
        if cropped_img is not None and cropped_img.size > 0:
            st.image(
                cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )
        else:
            st.warning("ยังไม่พบพื้นที่ ROI")

    # ------------------------------------------------------
    # 6. Detailed Analysis Metrics
    # ------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 สรุปผลการจำแนกประเภทและวัดมุมระดับสูง")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(label="มุมสุทธิที่วัดได้ (Hybrid Angle)", value=f"{final_angle}°")
    with m2:
        st.metric(label="ประเภทโมเดล (Gemini Classification)", value=model_name)
    with m3:
        st.metric(label="ระดับความเชื่อมั่น (Confidence)", value=f"{confidence}%")

    if gemini_result:
        st.success(f"💡 **การวิเคราะห์จาก Gemini AI:** {reasoning}")
        if features:
            st.markdown(
                "**ลักษณะทางโครงสร้างที่ตรวจพบ:** "
                + ", ".join([f"`{f}`" for f in features])
            )
else:
    st.info("💡 อัปโหลดรูปภาพและใส่ API Key ด้านซ้ายเพื่อเริ่มประมวลผลระบบ Hybrid AI")
