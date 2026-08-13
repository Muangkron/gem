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
    page_title="ระบบประเมินความหวานสับปะรดภูเก็ต",
    page_icon="🍍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🍍 ระบบประเมินความหวานสับปะรดภูเก็ต (°Brix Estimator)")
st.caption(
    "วิเคราะห์รูปแบบเกลียวตาด้วย Gemini AI Vision พร้อมเครื่องมือทาบเส้นมุมเรียลไทม์"
)

# -----------------------------------------------------------------------------
# 2. การจัดการ API Key & Gemini Model Version
# -----------------------------------------------------------------------------
api_key = st.secrets.get("GEMINI_API_KEY", "")

with st.sidebar:
    st.header("⚙️ การตั้งค่าระบบ")
    if not api_key:
        api_key = st.text_input("กรอก Gemini API Key:", type="password")

    gemini_model_name = st.selectbox(
        "เลือกเลือกรุ่นโมเดล Gemini:",
        ("gemini-2.5-flash", "gemini-1.5-flash", "gemini-2.0-flash"),
        index=0,
    )

    st.markdown("---")
    st.markdown("### 📐 สมการอ้างอิงโครงงาน")
    st.info(
        "**Model 5-8-13 (ผลเล็ก-กลาง / ตาห่าง):**\n"
        "- มุมอุดมคติ $(\\theta_0) = 155^\\circ$\n"
        "- $\\text{Brix} = -0.0196x^2 + 0.0045x + 16.757$\n\n"
        "**Model 8-13-21 (ผลกลาง-ใหญ่ / ตาแน่น):**\n"
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
# 4. ฟังก์ชัน AI Vision วิเคราะห์ร่องตาและเลือกโมเดลอย่างแม่นยำ
# -----------------------------------------------------------------------------
def analyze_pineapple_with_ai(pil_img, key, model_name):
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel(model_name)

        prompt = """
        คุณคือผู้เชี่ยวชาญด้านเรขาคณิตตาสับปะรดและ Computer Vision

        จงวิเคราะห์ภาพถ่ายสับปะรดนี้ (ตัดใบและฉากหลังออก สแกนเฉพาะส่วนผลตรงกลาง):

        1. **การจำแนกโมเดล (Model Classification):**
           - สังเกตความหนาแน่นและสัดส่วนจำนวนเกลียวตาในแกนหลัก:
             * หากตาอัดแน่นถี่สูง / สัดส่วนเกลียวตา 8-13-21 -> เลือก "Model 8-13-21"
             * หากตาห่างกว่า / สัดส่วนเกลียวตา 5-8-13 -> เลือก "Model 5-8-13"

        2. **การวัดมุมแหลมเกลียวตา (Acute Angle Measurement - ϕ):**
           - สังเกตแนวร่องตาหลักที่เฉียงจาก "ซ้ายบน" ลงไป "ขวาล่าง" บริเวณตรงกลางผล
           - ประเมินมุมแหลม ϕ (phi) ที่แนวเส้นร่องตานี้ทำกับแนวเส้นแนวนอนฝั่งขวา (โดยปกติมุม ϕ จะอยู่ช่วง 25.0° ถึง 60.0°)

        ตอบกลับเฉพาะข้อความ JSON เท่านั้น (ห้ามมีคำเกริ่นหรือ Markdown Fence):
        {
          "selected_model": "Model 8-13-21 หรือ Model 5-8-13",
          "acute_angle_phi": 41.5,
          "reasoning": "อธิบายสั้นๆ ว่าเห็นร่องตาและรูปแบบเกลียวอย่างไร"
        }
        """

        response = model.generate_content([prompt, pil_img])
        json_match = re.search(r"\{.*\}", response.text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            return (
                data.get("selected_model", "Model 8-13-21"),
                float(data.get("acute_angle_phi", 41.0)),
                data.get("reasoning", "วิเคราะห์สรีระตาสำเร็จ"),
            )
    except Exception as e:
        st.warning(f"การสแกนด้วย AI ขัดข้อง: {str(e)} (ใช้ค่าเริ่มต้นชั่วคราว)")

    return "Model 8-13-21", 41.0, "ใช้ค่าอ้างอิงมาตรฐาน"


# -----------------------------------------------------------------------------
# 5. ฟังก์ชันวาดเส้น Overlay อ้างอิงมุมลงบนภาพ
# -----------------------------------------------------------------------------
def draw_angle_overlay(pil_img, phi_deg):
    img_copy = pil_img.copy()
    draw = PIL.ImageDraw.Draw(img_copy)
    width, height = img_copy.size

    # กำหนดจุดศูนย์กลางบริเวณผลสับปะรด
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
# 6. ส่วนการทำงานหลัก (Main UI)
# -----------------------------------------------------------------------------
col_left, col_right = st.columns([1.1, 0.9])

with col_left:
    st.subheader("📷 1. อัปโหลดรูปถ่ายสับปะรด")
    uploaded_file = st.file_uploader(
        "เลือกไฟล์ภาพสับปะรด",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded_file is not None:
        image = PIL.Image.open(uploaded_file)

        # ทำการประมวลผลด้วย AI Vision เพียงครั้งเดียวเมื่ออัปโหลดภาพใหม่
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if (
            "current_file_id" not in st.session_state
            or st.session_state.current_file_id != file_id
        ):
            st.session_state.current_file_id = file_id

            if api_key:
                with st.spinner("🤖 AI กำลังสแกนลักษณะตาและวัดมุมเกลียว..."):
                    detected_model, detected_phi, reasoning = (
                        analyze_pineapple_with_ai(
                            image, api_key, gemini_model_name
                        )
                    )
            else:
                detected_model, detected_phi, reasoning = (
                    "Model 8-13-21",
                    41.0,
                    "กรุณากรอก API Key เพื่อสแกนอัตโนมัติ",
                )

            st.session_state.detected_phi = detected_phi
            st.session_state.detected_model = detected_model
            st.session_state.reasoning = reasoning

        st.markdown("---")
        st.subheader("🎛️ 2. เครื่องมือปรับทาบเส้นมุม (Manual Fine-Tune)")
        st.caption(
            "ระบบ AI สแกนค่าเริ่มต้นให้แล้ว หากเส้นสีแดงส้มยังไม่ทาบตรงร่องตาจริง คุณสามารถลาก Slider เพื่อหมุนเส้นให้ตรงเป๊ะได้ทันที"
        )

        # Slider ปรับมุม phi (ดึงค่าเริ่มต้นมาจาก AI)
        manual_phi = st.slider(
            "ปรับมุมแหลม ($\phi$) ที่เส้นเกลียวทำกับแนวนอนฝั่งขวา:",
            min_value=15.0,
            max_value=75.0,
            value=float(st.session_state.detected_phi),
            step=0.5,
        )

        # Radio เลือกโมเดล (ดึงค่าเริ่มต้นมาจาก AI)
        default_index = (
            0
            if st.session_state.detected_model == "Model 8-13-21"
            else 1
        )
        selected_model = st.radio(
            "เลือกแบบจำลองสับปะรด:",
            ("Model 8-13-21", "Model 5-8-13"),
            index=default_index,
        )

        calc_theta = 180.0 - manual_phi

        # วาดเส้น Overlay ตามมุมที่ขยับบน Slider เรียลไทม์
        overlay_img = draw_angle_overlay(image, manual_phi)
        st.image(
            overlay_img,
            caption=f"เส้นสีฟ้า (Baseline 0°) | เส้นสีแดงส้ม (แนวเกลียวตา ϕ = {manual_phi:.1f}°) | กรอบสีเหลือง (Center ROI)",
            use_container_width=True,
        )

with col_right:
    st.subheader("📊 3. ผลการคำนวณและประเมินค่า Brix")

    if uploaded_file is not None:
        ideal_angle, x_val, brix_val = calculate_brix(
            calc_theta, selected_model
        )

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
        st.markdown("### 📝 ข้อมูลการวิเคราะห์จาก AI")
        st.info(f"**AI Reasoning:** {st.session_state.get('reasoning', '-')}")

        st.json(
            {
                "Selected Model": selected_model,
                "Ideal Angle (theta_0)": f"{ideal_angle:.1f}°",
                "Detected Acute Angle (phi)": f"{manual_phi:.1f}°",
                "Calculated Spiral Angle (theta)": f"{calc_theta:.1f}°",
                "Deviation Value (x)": f"{x_val:.4f}",
                "Estimated Sweetness": f"{brix_val:.2f} °Brix",
            }
        )
    else:
        st.info("👈 กรุณาอัปโหลดรูปถ่ายสับปะรดที่เมนูด้านซ้ายเพื่อเริ่มต้นวิเคราะห์")
