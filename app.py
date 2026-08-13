import json
import math
import re
import google.generativeai as genai
import PIL.Image
import PIL.ImageDraw
import streamlit as st

# -----------------------------------------------------------------------------
# 1. ตั้งค่าโครงสร้างหน้าเว็บ (Page Configuration)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ระบบประเมินความหวานสับปะรดภูเก็ต (°Brix Estimator)",
    page_icon="🍍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🍍 ระบบประเมินความหวานสับปะรดภูเก็ต (°Brix Estimator)")
st.caption(
    "วิเคราะห์จำแนกโมเดลสับปะรดขั้นสูงด้วย Gemini 3.6 ร่วมกับระบบวัดมุมเกลียวเรขาคณิต"
)

# ล็อกรุ่นโมเดล Gemini 3.6 ในหลังบ้านทันที
GEMINI_MODEL_VERSION = "gemini-3.6-flash"

# -----------------------------------------------------------------------------
# 2. การจัดการ API Key & Sidebar Reference
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
# 3. ฟังก์ชันคำนวณความหวานตามแบบจำลองคณิตศาสตร์
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
# 4. ฟังก์ชัน Gemini 3.6 วิเคราะห์กายภาพขั้นสูง (Chain-of-Thought Classification)
# -----------------------------------------------------------------------------
def analyze_pineapple_high_precision(pil_img, key):
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel(GEMINI_MODEL_VERSION)

        prompt = """
        คุณคือระบบ AI ผู้เชี่ยวชาญทางพฤกษศาสตร์และเรขาคณิตจำแนกสับปะรดภูเก็ต (Queen Pineapple)

        จงสแกนภาพถ่ายสับปะรดนี้ (มุ่งเน้นสแกนผิวผลตรงกลาง ละเว้นจุกและฉากหลัง) แล้ววิเคราะห์ดัชนีทางกายภาพทีละขั้นตอน:

        [ขั้นตอนการวิเคราะห์เชิงโครงสร้าง (Morphological Metrics)]:
        1. **Fruit Shape Ratio (ทรงผล):** ทรงกระบอกยาว (Cylindrical/Elongated) หรือ ทรงกลมป้อม (Ovoid/Plump)
        2. **Eye Size Scale (ขนาดตา):** ตาค่อนข้างใหญ่มีเนื้อที่กว้าง หรือ ตาขนาดเล็กอัดแน่น
        3. **Horizontal Eye Count in 2D View (จำลองการนับจานตาแนวแนวนอนที่เห็นในรูป):**
           - หากมองเห็นตาเรียงตามแนวแนวนอนประมาณ 4 ถึง 6 ตา -> เป็นลักษณะของลำดับเกลียว 5-8-13
           - หากมองเห็นตาเรียงอัดแน่นตามแนวแนวนอนประมาณ 7 ถึง 10+ ตา -> เป็นลักษณะของลำดับเกลียว 8-13-21
        4. **Spiral Pitch Density (ความถี่เกลียว):** ความถี่ต่ำร่องห่าง หรือ ความถี่สูงเกลียวชิดอัดแน่น

        [การตัดสินจำแนกโมเดลอย่างเด็ดขาด (Strict Classification Rule)]:
        - หากเป็น ทรงกระบอกยาว/ตาใหญ่ห่าง/นับตาแนวแนวนอนได้ประมาณ 4-6 ตา -> เลือก "Model 5-8-13"
        - หากเป็น ทรงกลมป้อม/ตาเล็กอัดถี่แน่น/นับตาแนวแนวนอนได้ตั้งแต่ 7 ตาขึ้นไป -> เลือก "Model 8-13-21"

        [การวัดมุมเกลียวหลัก (Acute Angle Measurement - ϕ)]:
        - มองร่องเกลียวตาหลักที่เอียงจาก "ซ้ายบน" ลงไป "ขวาล่าง" บริเวณแกนกลางผล
        - ประเมินมุมแหลม ϕ (phi) ที่แนวเส้นร่องตานี้ทำกับเส้นแนวแนวนอนฝั่งขวา (ช่วงปกติ 25.0° ถึง 60.0°)

        ตอบกลับเฉพาะข้อความ JSON เท่านั้น (ห้ามมีคำเกริ่นหรือ Markdown Block):
        {
          "fruit_shape": "ทรงกระบอกยาว หรือ ทรงกลมป้อม",
          "eye_density_assessment": "ตาห่างความหนาแน่นน้อย หรือ ตาเล็กอัดแน่นถี่สูง",
          "horizontal_eyes_count_est": "4-6 ตา หรือ 7-10 ตา",
          "selected_model": "Model 5-8-13 หรือ Model 8-13-21",
          "acute_angle_phi": 41.5,
          "classification_reasoning": "อธิบายสั้นๆ ถึงสัดส่วนตาและทรงผลที่ทำให้เลือกโมเดลนี้"
        }
        """

        response = model.generate_content([prompt, pil_img])
        json_match = re.search(r"\{.*\}", response.text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            return (
                data.get("selected_model", "Model 8-13-21"),
                float(data.get("acute_angle_phi", 41.0)),
                data.get("fruit_shape", "ไม่ระบุ"),
                data.get("eye_density_assessment", "ไม่ระบุ"),
                data.get("horizontal_eyes_count_est", "ไม่ระบุ"),
                data.get("classification_reasoning", "วิเคราะห์กายภาพสำเร็จ"),
            )
    except Exception as e:
        st.warning(
            f"การสแกนด้วย Gemini 3.6 ขัดข้อง: {str(e)} (ใช้ค่าตั้งต้นระบบ)"
        )

    return (
        "Model 8-13-21",
        41.0,
        "ทรงกลมป้อม",
        "ตาเล็กอัดแน่นถี่สูง",
        "7-10 ตา",
        "ใช้ค่าอ้างอิงมาตรฐาน",
    )


# -----------------------------------------------------------------------------
# 5. ฟังก์ชันวาดเส้น Overlay อ้างอิงมุมลงบนภาพ (Visual Feedback)
# -----------------------------------------------------------------------------
def draw_angle_overlay(pil_img, phi_deg):
    img_copy = pil_img.copy()
    draw = PIL.ImageDraw.Draw(img_copy)
    width, height = img_copy.size

    cx, cy = width // 2, height // 2
    line_length = min(width, height) // 3

    # กรอบ Center ROI สีเหลืองประ
    rx1, ry1 = int(width * 0.3), int(height * 0.25)
    rx2, ry2 = int(width * 0.7), int(height * 0.75)
    draw.rectangle([rx1, ry1, rx2, ry2], outline="#FFD700", width=3)

    # 1. เส้นแนวนอน Baseline 0° (สีฟ้า)
    draw.line(
        [(cx - line_length, cy), (cx + line_length, cy)],
        fill="#00E5FF",
        width=4,
    )

    # 2. เส้นแนวเกลียวตาตามมุม phi (สีแดงส้ม)
    phi_rad = math.radians(phi_deg)
    dx = int(line_length * math.cos(phi_rad))
    dy = int(line_length * math.sin(phi_rad))

    p1 = (cx - dx, cy - dy)
    p2 = (cx + dx, cy + dy)
    draw.line([p1, p2], fill="#FF3D00", width=5)

    # จุดศูนย์กลางหมุน
    draw.ellipse(
        [cx - 6, cy - 6, cx + 6, cy + 6], fill="#FFFFFF", outline="#000000"
    )

    return img_copy


# -----------------------------------------------------------------------------
# 6. ส่วนการทำงานหลัก (Main Application UI)
# -----------------------------------------------------------------------------
col_left, col_right = st.columns([1.1, 0.9])

with col_left:
    st.subheader("📷 1. อัปโหลดรูปถ่ายสับปะรด")
    uploaded_file = st.file_uploader(
        "เลือกไฟล์ภาพสับปะรด (PNG, JPG, JPEG)",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded_file is not None:
        image = PIL.Image.open(uploaded_file)

        # สแกนด้วย Gemini 3.6 เพียงครั้งเดียวเมื่ออัปโหลดไฟล์ใหม่
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if (
            "current_file_id" not in st.session_state
            or st.session_state.current_file_id != file_id
        ):
            st.session_state.current_file_id = file_id

            if api_key:
                with st.spinner("🤖 Gemini 3.6 กำลังวิเคราะห์สรีระตาและแยกโมเดลเชิงลึก..."):
                    (
                        detected_model,
                        detected_phi,
                        fruit_shape,
                        eye_density,
                        eye_count,
                        reasoning,
                    ) = analyze_pineapple_high_precision(image, api_key)
            else:
                (
                    detected_model,
                    detected_phi,
                    fruit_shape,
                    eye_density,
                    eye_count,
                    reasoning,
                ) = (
                    "Model 8-13-21",
                    41.0,
                    "ไม่ระบุ",
                    "ไม่ระบุ",
                    "ไม่ระบุ",
                    "กรุณากรอก API Key ในแถบด้านซ้ายเพื่อสแกนอัตโนมัติ",
                )

            st.session_state.detected_phi = detected_phi
            st.session_state.detected_model = detected_model
            st.session_state.fruit_shape = fruit_shape
            st.session_state.eye_density = eye_density
            st.session_state.eye_count = eye_count
            st.session_state.reasoning = reasoning

        st.markdown("---")
        st.subheader("🎛️ 2. เครื่องมือปรับทาบเส้นมุม (Manual Fine-Tune)")
        st.caption(
            "หากต้องการปรับหมุนเส้นสีแดงส้มให้ทาบขนานกับร่องตาจริงเป๊ะๆ สามารถลาก Slider ด้านล่างได้ทันที"
        )

        # Slider ปรับมุม phi (ดึงค่าเริ่มต้นมาจาก Gemini 3.6)
        manual_phi = st.slider(
            "ปรับมุมแหลม ($\phi$) ที่เส้นเกลียวทำกับแนวนอนฝั่งขวา:",
            min_value=15.0,
            max_value=75.0,
            value=float(st.session_state.detected_phi),
            step=0.5,
        )

        # Radio เลือกโมเดล (ดึงค่าเริ่มต้นมาจาก Gemini 3.6 และเปิดให้แก้ไขได้หากต้องการ)
        default_index = (
            0 if st.session_state.detected_model == "Model 8-13-21" else 1
        )
        selected_model = st.radio(
            "แบบจำลองสับปะรด (จำแนกอัตโนมัติโดย Gemini 3.6):",
            ("Model 8-13-21", "Model 5-8-13"),
            index=default_index,
        )

        calc_theta = 180.0 - manual_phi

        # วาดเส้น Overlay ทับลงบนรูปภาพเรียลไทม์
        overlay_img = draw_angle_overlay(image, manual_phi)
        st.image(
            overlay_img,
            caption=f"เส้นสีฟ้า (Baseline 0°) | เส้นสีแดงส้ม (แนวเกลียวตา ϕ = {manual_phi:.1f}°) | กรอบสีเหลือง (Center ROI)",
            use_container_width=True,
        )

with col_right:
    st.subheader("📊 3. ผลการวิเคราะห์สรีระและประเมินค่า Brix")

    if uploaded_file is not None:
        ideal_angle, x_val, brix_val = calculate_brix(
            calc_theta, selected_model
        )

        # แสดงกล่องรายงานดัชนีกายภาพที่ AI สแกนได้
        st.markdown("### 🔍 ดัชนีทางกายภาพที่ตรวจจับได้ (Gemini 3.6)")
        c1, c2, c3 = st.columns(3)
        c1.metric("รูปทรงผล", st.session_state.get("fruit_shape", "-"))
        c2.metric("ความแน่นของตา", st.session_state.get("eye_density", "-"))
        c3.metric("ประมาณการตาในแนวระนาบ", st.session_state.get("eye_count", "-"))

        st.markdown("---")
        m1, m2 = st.columns(2)
        m1.metric("โมเดลสับปะรดที่ใช้งาน", selected_model)
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
        st.markdown("### 📝 เหตุผลเชิงจำแนกของ AI")
        st.info(f"**AI Reasoning:** {st.session_state.get('reasoning', '-')}")

        st.json(
            {
                "AI Vision Engine": GEMINI_MODEL_VERSION,
                "Selected Model": selected_model,
                "Fruit Shape": st.session_state.get("fruit_shape", "-"),
                "Eye Density": st.session_state.get("eye_density", "-"),
                "Horizontal Eye Count": st.session_state.get("eye_count", "-"),
                "Ideal Angle (theta_0)": f"{ideal_angle:.1f}°",
                "Detected Acute Angle (phi)": f"{manual_phi:.1f}°",
                "Calculated Spiral Angle (theta)": f"{calc_theta:.1f}°",
                "Deviation Value (x)": f"{x_val:.4f}",
                "Estimated Sweetness": f"{brix_val:.2f} °Brix",
            }
        )
    else:
        st.info("👈 กรุณาอัปโหลดรูปถ่ายสับปะรดที่เมนูด้านซ้ายเพื่อเริ่มต้นวิเคราะห์")
