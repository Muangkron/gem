import json
import math
import cv2
from google import genai
from google.genai import types
import numpy as np
from PIL import Image
import streamlit as st

# ==========================================================
# 1. Page Configuration
# ==========================================================
st.set_page_config(
    page_title="ระบบประเมินค่าความหวานสับปะรด (Brix) & มุมเกลียว",
    page_icon="🍍",
    layout="wide",
)

st.title("🍍 ระบบประเมินค่าความหวานสับปะรด (Brix Estimation System)")
st.caption(
    "วัดมุมเกลียว (Spiral Angle) จำแนกโมเดล (Model 5-8-13 / 8-13-21) และประเมินค่า Brix ด้วย Gemini 3.6 Flash + OpenCV"
)

# ==========================================================
# 2. Sidebar API Settings
# ==========================================================
st.sidebar.header("🔑 ตั้งค่า Gemini API")
api_key = st.sidebar.text_input(
    "Gemini API Key",
    value=st.secrets.get("GEMINI_API_KEY", ""),
    type="password",
    help="ใส่ API Key จาก Google AI Studio",
)


# ==========================================================
# 3. Core Brix Calculation & Model Logic
# ==========================================================
def calculate_brix_polynomial(angle, model_type):
    """คำนวณค่าความหวาน Brix จากมุมเกลียว (phi) และโมเดลสับปะรด

    ใช้สมการ Polynomial Regression
    """
    phi = float(angle)

    if "5-8-13" in model_type:
        # สมการสำหรับ Model 5-8-13
        brix = 0.0015 * (phi**2) + 0.12 * phi + 7.5
    elif "8-13-21" in model_type:
        # สมการสำหรับ Model 8-13-21
        brix = 0.0018 * (phi**2) + 0.15 * phi + 8.0
    else:
        # สมการมาตรฐานทั่วไป
        brix = 0.0016 * (phi**2) + 0.13 * phi + 7.8

    return round(float(brix), 2)


def detect_spiral_angle_opencv(image):
    """ตรวจจับมุมเกลียวและเส้นตาของสับปะรดด้วย OpenCV"""
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


def analyze_pineapple_gemini(pil_img, key):
    """ส่งภาพให้ Gemini 3.6 Flash วิเคราะห์มุมเกลียว จำแนกโมเดลสับปะรด และประเมินค่า Brix"""
    if not key:
        return None

    try:
        client = genai.Client(api_key=key)

        prompt = """
        Analyze this pineapple image for Brix sweetness estimation and spiral angle measurement:
        1. **Spiral Angle**: Determine the primary orientation/tilt angle of the pineapple eyes/grooves in degrees (-180.0 to 180.0).
        2. **Model Classification**: Identify the spiral sequence model based on eye density/pattern. Select STRICTLY between "Model 5-8-13" or "Model 8-13-21".
        3. **Bounding Box**: Provide normalized coordinates [ymin, xmin, ymax, xmax] (0 to 1000) around the pineapple fruit body.
        4. **Confidence**: Provide a confidence score percentage (0-100).

        Respond STRICTLY in JSON format:
        {
            "model_type": "Model 5-8-13",
            "spiral_angle": 32.5,
            "bounding_box_1000": [ymin, xmin, ymax, xmax],
            "confidence_score": 92.5,
            "reasoning": "Observed eye spiral density matches Fibonacci Model 5-8-13"
        }
        """

        # ⚡ เรียกใช้โมเดล gemini-3.6-flash
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=[pil_img, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )

        return json.loads(response.text)
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดจาก Gemini API: {str(e)}")
        return None


# ==========================================================
# 4. Main Application Workflow
# ==========================================================
uploaded_file = st.file_uploader(
    "อัปโหลดรูปภาพสับปะรดเพื่อเริ่มการวิเคราะห์ (JPG, PNG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    img_h, img_w = cv_image.shape[:2]

    # ประมวลผลพื้นฐานด้วย OpenCV
    cv_phi, cv_roi, edges = detect_spiral_angle_opencv(cv_image)

    # ปุ่มสั่งงาน Gemini AI
    st.markdown("---")
    if st.button("🚀 สั่ง Gemini 3.6 Flash วิเคราะห์สับปะรด (Analyze Pineapple)"):
        if not api_key:
            st.error("กรุณากรอก Gemini API Key ในแถบด้านซ้ายก่อนครับ")
        else:
            with st.spinner(
                "🧠 Gemini 3.6 Flash กำลังวัดมุมเกลียวและจำแนกโมเดลสับปะรด..."
            ):
                res = analyze_pineapple_gemini(pil_image, api_key)

                if res:
                    g_box = res.get(
                        "bounding_box_1000", [100, 100, 900, 900]
                    )
                    ymin = int((g_box[0] / 1000.0) * img_h)
                    xmin = int((g_box[1] / 1000.0) * img_w)
                    ymax = int((g_box[2] / 1000.0) * img_h)
                    xmax = int((g_box[3] / 1000.0) * img_w)

                    st.session_state["analyzed"] = True
                    st.session_state["model_type"] = res.get(
                        "model_type", "Model 5-8-13"
                    )
                    st.session_state["spiral_angle"] = float(
                        res.get("spiral_angle", cv_phi)
                    )
                    st.session_state["confidence"] = float(
                        res.get("confidence_score", 0.0)
                    )
                    st.session_state["reasoning"] = res.get("reasoning", "")

                    st.session_state["roi_xmin"] = max(0, xmin)
                    st.session_state["roi_ymin"] = max(0, ymin)
                    st.session_state["roi_xmax"] = min(img_w, xmax)
                    st.session_state["roi_ymax"] = min(img_h, ymax)
                    st.success("วิเคราะห์สำเร็จด้วย Gemini 3.6 Flash!")

    # ค่าเริ่มต้น Session State
    if "analyzed" not in st.session_state:
        st.session_state["analyzed"] = False
        st.session_state["model_type"] = "Model 5-8-13"
        st.session_state["spiral_angle"] = float(cv_phi)
        st.session_state["confidence"] = 0.0
        st.session_state["reasoning"] = "-"
        st.session_state["roi_xmin"] = (
            int(cv_roi[0]) if cv_roi else int(img_w * 0.1)
        )
        st.session_state["roi_ymin"] = (
            int(cv_roi[1]) if cv_roi else int(img_h * 0.1)
        )
        st.session_state["roi_xmax"] = (
            int(cv_roi[2]) if cv_roi else int(img_w * 0.9)
        )
        st.session_state["roi_ymax"] = (
            int(cv_roi[3]) if cv_roi else int(img_h * 0.9)
        )

    # ------------------------------------------------------
    # 5. Human-in-the-Loop Controls (ปรับค่าความหวาน & มุมเกลียว)
    # ------------------------------------------------------
    st.markdown("---")
    st.subheader(
        "🛠️ แผงปรับแต่งมุมเกลียวและแบบจำลอง (Human Fine-Tuning Panel)"
    )

    col_ctrl1, col_ctrl2 = st.columns(2)

    with col_ctrl1:
        st.markdown("##### 1. โมเดลและมุมเกลียวสับปะรด")
        model_options = ["Model 5-8-13", "Model 8-13-21", "Model Standard"]
        default_idx = (
            model_options.index(st.session_state["model_type"])
            if st.session_state["model_type"] in model_options
            else 0
        )

        user_model = st.selectbox(
            "จำแนกโมเดลสับปะรด", model_options, index=default_idx
        )
        user_angle = st.slider(
            "ปรับแก้มุมเกลียว (Spiral Angle - Degree)",
            min_value=-180.0,
            max_value=180.0,
            value=float(st.session_state["spiral_angle"]),
            step=0.1,
        )

    with col_ctrl2:
        st.markdown("##### 2. กรอบพิกัดสับปะรด (ROI)")
        box_c1, box_c2 = st.columns(2)
        with box_c1:
            user_xmin = st.number_input(
                "X Min",
                0,
                img_w,
                value=int(st.session_state["roi_xmin"]),
            )
            user_ymin = st.number_input(
                "Y Min",
                0,
                img_h,
                value=int(st.session_state["roi_ymin"]),
            )
        with box_c2:
            user_xmax = st.number_input(
                "X Max",
                0,
                img_w,
                value=int(st.session_state["roi_xmax"]),
            )
            user_ymax = st.number_input(
                "Y Max",
                0,
                img_h,
                value=int(st.session_state["roi_ymax"]),
            )

    # ------------------------------------------------------
    # 6. Real-time Brix Calculation & Rendering
    # ------------------------------------------------------
    calculated_brix = calculate_brix_polynomial(user_angle, user_model)

    annotated_img = cv_image.copy()
    cv2.rectangle(
        annotated_img,
        (user_xmin, user_ymin),
        (user_xmax, user_ymax),
        (0, 255, 0),
        3,
    )

    crop_x1 = max(0, min(user_xmin, user_xmax))
    crop_y1 = max(0, min(user_ymin, user_ymax))
    crop_x2 = min(img_w, max(user_xmin, user_xmax))
    crop_y2 = min(img_h, max(user_ymin, user_ymax))

    cropped_img = cv_image[crop_y1:crop_y2, crop_x1:crop_x2]

    st.markdown("---")
    disp_col1, disp_col2, disp_col3 = st.columns(3)

    with disp_col1:
        st.subheader("1. ภาพอ้างอิง + ROI")
        st.image(
            cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
            use_container_width=True,
        )

    with disp_col2:
        st.subheader("2. แผนผังขอบเกลียว (Edges)")
        if edges is not None:
            st.image(edges, use_container_width=True)

    with disp_col3:
        st.subheader("3. Auto-Crop ผลสับปะรด")
        if cropped_img.size > 0:
            st.image(
                cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )

    # ------------------------------------------------------
    # 7. Brix Metrics Display
    # ------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 ผลสรุปประเมินค่าความหวาน Brix และมุมเกลียว")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            label="ค่าความหวานประเมิน (°Brix)",
            value=f"{calculated_brix} °Brix",
        )
    with m2:
        st.metric(label="มุมเกลียวสรุป (Spiral Angle)", value=f"{user_angle:.1f}°")
    with m3:
        st.metric(label="แบบจำลองโมเดล", value=user_model)
    with m4:
        st.metric(
            label="ความน่าเชื่อถือ AI",
            value=f"{st.session_state['confidence']}%",
        )

    if st.session_state["reasoning"] != "-":
        st.info(
            f"💡 **เหตุผลการวิเคราะห์จาก Gemini 3.6 Flash:** {st.session_state['reasoning']}"
        )
else:
    st.info("💡 กรุณาอัปโหลดรูปภาพสับปะรดเพื่อเริ่มวิเคราะห์ค่าความหวาน Brix")
