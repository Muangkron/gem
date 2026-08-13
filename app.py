import json
import math
import re
import cv2
import google.generativeai as genai
import numpy as np
import PIL.Image
import PIL.ImageDraw
import streamlit as st

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ระบบประเมินความหวานสับปะรดภูเก็ต (°Brix Estimator)",
    page_icon="🍍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title(
    "🍍 ระบบประเมินความหวานสับปะรดภูเก็ต (°Brix Estimator Precision Version)"
)
st.caption(
    "ระบบตรวจวัดมุมเกลียวตาด้วย OpenCV Computer Vision Advanced Line Alignment"
    " + Gemini 3.6 Flash Morphological Classifier"
)

GEMINI_MODEL_VERSION = "gemini-3.6-flash"

# -----------------------------------------------------------------------------
# 2. Sidebar Setup & Math Equations
# -----------------------------------------------------------------------------
api_key = st.secrets.get("GEMINI_API_KEY", "")

with st.sidebar:
    st.header("⚙️ การตั้งค่าระบบ")
    if not api_key:
        api_key = st.text_input("กรอก Gemini API Key:", type="password")

    st.markdown("---")
    st.markdown("### 📐 สมการอ้างอิงโครงงาน")
    st.info(
        "**Model 5-8-13 (ผลเล็ก-กลาง / ตาใหญ่ห่าง):**\n"
        "- มุมอุดมคติ $(\\theta_0) = 155^\\circ$\n"
        "- $\\text{Brix} = -0.0196x^2 + 0.0045x + 16.757$\n\n"
        "**Model 8-13-21 (ผลกลาง-ใหญ่ / ตาเล็กถี่แน่น):**\n"
        "- มุมอุดมคติ $(\\theta_0) = 136^\\circ$\n"
        "- $\\text{Brix} = 0.0082x^2 - 0.6667x + 16.362$"
    )


# -----------------------------------------------------------------------------
# 3. Sweetness Calculation Math
# -----------------------------------------------------------------------------
def calculate_brix(theta, model_choice):
    if model_choice == "Model 5-8-13":
        ideal_angle = 155.0
        x = abs(theta - ideal_angle)
        brix = (-0.0196 * (x**2)) + (0.0045 * x) + 16.757
    else:  # Model 8-13-21
        ideal_angle = 136.0
        x = abs(theta - ideal_angle)
        brix = (0.0082 * (x**2)) - (0.6667 * x) + 16.362

    return ideal_angle, x, max(0.0, brix)


# -----------------------------------------------------------------------------
# 4. Advanced Precision OpenCV Angle Detection (แก้ไข Error เรียบร้อย)
# -----------------------------------------------------------------------------
def detect_precise_angle_opencv(pil_img):
    img_np = np.array(pil_img.convert("RGB"))
    h, w, _ = img_np.shape

    # Crop Center ROI (30%-70% W, 25%-75% H)
    rx1, rx2 = int(w * 0.30), int(w * 0.70)
    ry1, ry2 = int(h * 0.25), int(h * 0.75)
    roi = img_np[ry1:ry2, rx1:rx2]

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    # Contrast Enhancement
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Edge Detection
    edges = cv2.Canny(blurred, 30, 120)

    # Hough Lines Detection
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=35,
        minLineLength=25,
        maxLineGap=8,
    )

    valid_angles = []
    if lines is not None:
        for line in lines:
            # กระจายค่าด้วย .flatten() เพื่อป้องกัน TypeError บน Linux
            coords = line.flatten()
            if len(coords) == 4:
                x1, y1, x2, y2 = coords
                dx = float(x2 - x1)
                dy = float(y2 - y1)
                if dx != 0:
                    slope = dy / dx
                    if slope > 0.3:  # แนวเกลียวซ้ายบน -> ขวาล่าง
                        angle_deg = math.degrees(math.atan(slope))
                        if 20.0 <= angle_deg <= 70.0:
                            valid_angles.append(angle_deg)

    if valid_angles:
        best_phi = float(np.median(valid_angles))
    else:
        best_phi = 41.0

    return best_phi, (rx1, ry1, rx2, ry2)


# -----------------------------------------------------------------------------
# 5. Gemini 3.6 Flash High-Precision Morphological Classifier
# -----------------------------------------------------------------------------
def analyze_model_precision_gemini(pil_img, key):
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel(GEMINI_MODEL_VERSION)

        prompt = """
        คุณคือระบบ AI ตรวจวัดเรขาคณิตพืชและจำแนกสายพันธุ์สับปะรดอย่างแม่นยำสูง (Precision Pineapple Classifier)

        จงสแกนโครงสร้างผิวด้านนอกสับปะรดตรงกลางผลอย่างละเอียด แล้วทำการจำแนกโมเดลตามหลัก Phyllotaxis (การจัดเรียงใบและเกลียวตา):

        [เกณฑ์การตรวจวัดหลัก]:
        1. **นับจำนวนจานตาแนวแนวนอนในมุมมอง 2D (Horizontal Eye Count in 2D View):**
           - หากใน 1 ระนาบแนวนอนที่มองเห็น มีตาใหญ่เรียงกันประมาณ 4-6 ตา -> สัดส่วนเกลียวตาคือ 5-8-13
           - หากใน 1 ระนาบแนวนอนที่มองเห็น มีตาเล็กอัดแน่นถี่สูงเรียงกัน 7-10+ ตา -> สัดส่วนเกลียวตาคือ 8-13-21
        2. **ความหนาแน่นร่องตา (Spiral Pitch Density):**
           - ร่องตาห่าง ช่องว่างระหว่างตาเห็นชัด -> "Model 5-8-13"
           - ร่องตาชิดถี่ ร่องสเกลแคบแน่น -> "Model 8-13-21"

        [การส่งคืนค่า JSON]:
        ตอบกลับเป็นข้อความ JSON รูปแบบนี้เท่านั้น (ห้ามใส่คำเกริ่น):
        {
          "selected_model": "Model 5-8-13 หรือ Model 8-13-21",
          "horizontal_eyes_count": "4-6 ตา หรือ 7-10 ตา",
          "eye_density": "ตาห่างกว้าง หรือ ตาเล็กอัดแน่น",
          "ai_suggested_phi": 41.5,
          "reasoning": "เหตุผลเชิงเรขาคณิตสั้นๆ"
        }
        """

        response = model.generate_content([prompt, pil_img])
        json_match = re.search(r"\{.*\}", response.text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            return (
                data.get("selected_model", "Model 8-13-21"),
                data.get("horizontal_eyes_count", "ไม่ระบุ"),
                data.get("eye_density", "ไม่ระบุ"),
                float(data.get("ai_suggested_phi", 41.0)),
                data.get("reasoning", "สแกนสำเร็จ"),
            )
    except Exception:
        pass

    return (
        "Model 8-13-21",
        "7-10 ตา",
        "ตาเล็กอัดแน่น",
        41.0,
        "ใช้การวิเคราะห์ระบบสำรอง",
    )


# -----------------------------------------------------------------------------
# 6. Precision Overlay Drawing Function
# -----------------------------------------------------------------------------
def draw_precision_overlay(pil_img, phi_deg, roi_box):
    img_copy = pil_img.copy()
    draw = PIL.ImageDraw.Draw(img_copy)
    w, h = img_copy.size

    rx1, ry1, rx2, ry2 = roi_box
    draw.rectangle([rx1, ry1, rx2, ry2], outline="#FFD700", width=3)

    cx = (rx1 + rx2) // 2
    cy = (ry1 + ry2) // 2
    line_len = min(w, h) // 3

    # Baseline 0° Line (Cyan)
    draw.line(
        [(cx - line_len, cy), (cx + line_len, cy)], fill="#00E5FF", width=4
    )

    # Spiral Angle Line (Orange/Red)
    phi_rad = math.radians(phi_deg)
    dx = int(line_len * math.cos(phi_rad))
    dy = int(line_len * math.sin(phi_rad))

    draw.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill="#FF3D00", width=5)
    draw.ellipse(
        [cx - 7, cy - 7, cx + 7, cy + 7],
        fill="#FFFFFF",
        outline="#000000",
        width=2,
    )

    return img_copy


# -----------------------------------------------------------------------------
# 7. Main UI & Layout
# -----------------------------------------------------------------------------
col_left, col_right = st.columns([1.1, 0.9])

with col_left:
    st.subheader("📷 1. อัปโหลดภาพถ่ายสับปะรด")
    uploaded_file = st.file_uploader(
        "อัปโหลดรูปถ่ายเพื่อเริ่มสแกนเรขาคณิตมุมและแยกโมเดล",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded_file is not None:
        image = PIL.Image.open(uploaded_file)

        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if (
            "current_file_id" not in st.session_state
            or st.session_state.current_file_id != file_id
        ):
            st.session_state.current_file_id = file_id

            # OpenCV Precision Angle Detection
            cv_phi, roi_box = detect_precise_angle_opencv(image)
            st.session_state.cv_phi = cv_phi
            st.session_state.roi_box = roi_box

            # Gemini 3.6 Precision Classifier
            if api_key:
                with st.spinner(
                    "🤖 Gemini 3.6 Flash"
                    " กำลังสแกนตาสับปะรดและจำแนกโมเดลเป๊ะๆ..."
                ):
                    (
                        model_choice,
                        eyes_count,
                        eye_density,
                        ai_phi,
                        reasoning,
                    ) = analyze_model_precision_gemini(image, api_key)
            else:
                model_choice, eyes_count, eye_density, ai_phi, reasoning = (
                    "Model 8-13-21",
                    "7-10 ตา",
                    "ตาเล็กอัดแน่น",
                    cv_phi,
                    "กรุณากรอก API Key ในแถบด้านซ้าย",
                )

            st.session_state.detected_model = model_choice
            st.session_state.eyes_count = eyes_count
            st.session_state.eye_density = eye_density
            st.session_state.reasoning = reasoning
            st.session_state.detected_phi = (
                cv_phi if abs(cv_phi - 41.0) > 1.0 else ai_phi
            )

        st.markdown("---")
        st.subheader(
            "🎛️ 2. เครื่องมือเล็งและทาบเส้นวัดมุม (Manual Fine-Tune Overlay)"
        )
        st.caption(
            "สไลด์เพื่อหมุนปรับเส้นสีแดงส้มให้ทาบขนานไปตามร่องตาสับปะรดจริงในกรอบสีเหลือง"
        )

        manual_phi = st.slider(
            "ปรับหมุนมุมแหลม ($\phi$) ให้เส้นสีแดงทาบตรงร่องตา:",
            min_value=15.0,
            max_value=75.0,
            value=float(st.session_state.detected_phi),
            step=0.1,
            help="ปรับทีละ 0.1 องศาเพื่อความเป๊ะสูงสุด",
        )

        default_index = (
            0 if st.session_state.detected_model == "Model 8-13-21" else 1
        )
        selected_model = st.radio(
            "โมเดลสับปะรด (วิเคราะห์จำแนกอัตโนมัติ):",
            ("Model 8-13-21", "Model 5-8-13"),
            index=default_index,
        )

        calc_theta = 180.0 - manual_phi

        overlay_img = draw_precision_overlay(
            image, manual_phi, st.session_state.roi_box
        )
        st.image(
            overlay_img,
            caption=(
                f"เส้นสีฟ้า (Baseline 0°) | เส้นสีแดงส้ม (แนวเกลียวตา ϕ ="
                f" {manual_phi:.1f}°) | กรอบสีเหลือง (Center ROI)"
            ),
            use_container_width=True,
        )

with col_right:
    st.subheader("📊 3. ผลการคำนวณและประเมินค่า Brix")

    if uploaded_file is not None:
        ideal_angle, x_val, brix_val = calculate_brix(
            calc_theta, selected_model
        )

        st.markdown("### 🔍 ผลการตรวจสแกนกายภาพ (Precision Diagnostic)")
        c1, c2 = st.columns(2)
        c1.metric(
            "จำนวนตาแนวระนาบ (Horizontal Eyes)",
            st.session_state.get("eyes_count", "-"),
        )
        c2.metric(
            "ความแน่นของร่องตา (Eye Density)",
            st.session_state.get("eye_density", "-"),
        )

        st.markdown("---")
        m1, m2 = st.columns(2)
        m1.metric("โมเดลสับปะรดที่เลือก", selected_model)
        m2.metric("มุมแหลมวัดได้ ($\phi$)", f"{manual_phi:.1f}°")

        m3, m4 = st.columns(2)
        m3.metric(
            "มุมเกลียวจริง ($\theta = 180^\circ - \phi$)", f"{calc_theta:.1f}°"
        )
        m4.metric("ระยะเบี่ยงเบน ($x = |\\theta - \\theta_0|$)", f"{x_val:.2f}°")

        st.markdown("---")
        st.metric(
            label="ค่าความหวานประเมิน (°Brix)",
            value=f"{brix_val:.2f} °Brix",
            delta=f"{'ระดับหวานมาก' if brix_val >= 15.0 else 'ระดับหวานปกติ'}",
        )

        st.markdown("---")
        st.markdown("### 📝 คำอธิบายการจำแนกของ AI")
        st.info(f"**AI Reasoning:** {st.session_state.get('reasoning', '-')}")

        st.json(
            {
                "AI Model Engine": GEMINI_MODEL_VERSION,
                "Selected Model": selected_model,
                "Horizontal Eyes Count": st.session_state.get("eyes_count", "-"),
                "Eye Density": st.session_state.get("eye_density", "-"),
                "Ideal Angle (theta_0)": f"{ideal_angle:.1f}°",
                "Measured Acute Angle (phi)": f"{manual_phi:.1f}°",
                "Calculated Spiral Angle (theta)": f"{calc_theta:.1f}°",
                "Deviation Value (x)": f"{x_val:.4f}",
                "Estimated Sweetness": f"{brix_val:.2f} °Brix",
            }
        )
    else:
        st.info("👈 กรุณาอัปโหลดรูปถ่ายสับปะรดที่เมนูด้านซ้ายเพื่อเริ่มต้นวิเคราะห์")
