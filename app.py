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
    page_title="ระบบประเมินความหวานสับปะรดอัตโนมัติ (AI Brix Estimator)",
    page_icon="🍍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🍍 ระบบประเมินความหวานสับปะรดภูเก็ตอัตโนมัติ 100% (Zero-Click AI)")
st.caption(
    "เพียงอัปโหลดรูปถ่ายสับปะรด ระบบจะวิเคราะห์รูปแบบเกลียวตา วัดมุมเรขาคณิต และคำนวณค่า Brix ให้อัตโนมัติ"
)

# -----------------------------------------------------------------------------
# 2. การจัดการ API Key
# -----------------------------------------------------------------------------
api_key = st.secrets.get("GEMINI_API_KEY", "")

with st.sidebar:
    st.header("⚙️ การตั้งค่าระบบ")
    if not api_key:
        api_key = st.text_input("ใส่ Gemini API Key ของคุณ:", type="password")

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
# 4. ฟังก์ชันวาดเส้น Overlay ทับบนรูปภาพ (Visual Feedback)
# -----------------------------------------------------------------------------
def draw_angle_overlay(pil_img, phi_deg):
    img_copy = pil_img.copy()
    draw = PIL.ImageDraw.Draw(img_copy)
    width, height = img_copy.size

    # จุดศูนย์กลางผลสับปะรด
    cx, cy = width // 2, height // 2
    line_length = min(width, height) // 3

    # กรอบ Center ROI (30%-70% Width, 25%-75% Height)
    draw.rectangle(
        [width * 0.3, height * 0.25, width * 0.7, height * 0.75],
        outline="#FFD700",
        width=3,
    )

    # 1. เส้นแนวนอนอ้างอิง Baseline 0° (สีฟ้า)
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

    # วาดจุดหมุนตรงกลาง
    draw.ellipse(
        [cx - 6, cy - 6, cx + 6, cy + 6], fill="#FFFFFF", outline="#000000"
    )

    return img_copy


# -----------------------------------------------------------------------------
# 5. ส่วนรับภาพและประมวลผลอัตโนมัติหลัก
# -----------------------------------------------------------------------------
col_left, col_right = st.columns([1.1, 0.9])

with col_left:
    st.subheader("📷 1. อัปโหลดรูปถ่ายสับปะรด")
    uploaded_file = st.file_uploader(
        "เลือกไฟล์ภาพสับปะรด (ระบบจะประมวลผลให้อัตโนมัติทันที)",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded_file is not None:
        image = PIL.Image.open(uploaded_file)

        if not api_key:
            st.warning("⚠️ กรุณากรอก Gemini API Key ในแถบด้านซ้ายก่อนครับ")
        else:
            with st.spinner("🤖 Gemini AI กำลังวิเคราะห์รูปภาพและวัดมุมให้อัตโนมัติ..."):
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel("gemini-2.5-flash")

                    # Prompt สำหรับวิเคราะห์ภาพและตรวจจับอัตโนมัติ
                    prompt = """
                    คุณคือระบบ Computer Vision และ AI วิเคราะห์โครงสร้างสับปะรดอัตโนมัติ

                    จงประมวลผลภาพสับปะรดนี้และตอบกลับเป็น JSON ตามขั้นตอนอย่างเคร่งครัด:

                    1. **จำแนกแบบจำลองสับปะรด (Model Classification):**
                       - ตรวจประเมินจากความแน่นของตาและทรงผล (ห้ามนำมุมองศามาคิดในการเลือกโมเดล):
                         * หากเป็นสับปะรดทรงกลมป้อม / ตาถี่แน่นมาก (กลุ่มเกลียว 8-13-21) -> เลือก "Model 8-13-21"
                         * หากเป็นสับปะรดทรงยาว / ตาห่าง ความหนาแน่นตาน้อย (กลุ่มเกลียว 5-8-13) -> เลือก "Model 5-8-13"

                    2. **วัดมุมแนวเกลียวตาแกนกลางผล (Angle Measurement):**
                       - มองเฉพาะบริเวณแกนกลางผล (Center Region)
                       - ลากเส้นตามแนวเกลียวตาสับปะรดหลักจาก ซ้ายบน -> ขวาล่าง
                       - วัดมุมแหลม ϕ (phi) ในหน่วยองศา ที่เส้นเกลียวทำกับเส้นแนวนอนฝั่งขวา (โดยปกติจะอยู่ช่วง 20.0° - 65.0°)

                    ตอบกลับเฉพาะข้อความ JSON รูปแบบนี้เท่านั้น (ห้ามมีคำเกริ่นหรือ Markdown Fence):
                    {
                      "selected_model": "Model 8-13-21 หรือ Model 5-8-13",
                      "acute_angle_phi": 41.0,
                      "reasoning": "อธิบายสั้นๆ ว่าประเมินลักษณะกายภาพอย่างไรจึงเลือกโมเดลนั้น และพบมุมแหลม ϕ กี่องศา"
                    }
                    """

                    response = model.generate_content([prompt, image])
                    raw_text = response.text

                    # ดึง JSON จากผลลัพธ์ AI
                    json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)

                    if json_match:
                        data = json.loads(json_match.group(0))

                        auto_model = data.get("selected_model", "Model 8-13-21")
                        auto_phi = float(data.get("acute_angle_phi", 41.0))
                        reasoning = data.get("reasoning", "")

                        # คำนวณมุมจริง θ
                        auto_theta = 180.0 - auto_phi

                        # คำนวณค่า Brix จากสมการหลังบ้าน
                        ideal_angle, x_val, brix_val = calculate_brix(
                            auto_theta, auto_model
                        )

                        # วาดเส้น Overlay
                        overlay_img = draw_angle_overlay(image, auto_phi)

                        st.image(
                            overlay_img,
                            caption=f"วิเคราะห์อัตโนมัติ: {auto_model} | เส้นสีฟ้า (Baseline) | เส้นสีแดงส้ม (แนวเกลียวตา ϕ = {auto_phi:.1f}°)",
                            use_container_width=True,
                        )

                    else:
                        st.error("เกิดข้อผิดพลาดในการแปลงผลลัพธ์จาก AI")

                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาด: {str(e)}")

with col_right:
    st.subheader("📊 2. ผลการประเมินความหวาน (°Brix)")

    if uploaded_file is not None and api_key and 'brix_val' in locals():
        st.success("✅ วิเคราะห์และคำนวณผลเสร็จสิ้นอัตโนมัติ!")

        m1, m2 = st.columns(2)
        m1.metric("แบบจำลองที่ AI เลือก", auto_model)
        m2.metric("มุมแหลมที่ AI วัดได้ ($\phi$)", f"{auto_phi:.1f}°")

        m3, m4 = st.columns(2)
        m3.metric(
            "มุมเกลียวจริง ($\theta = 180^\circ - \phi$)", f"{auto_theta:.1f}°"
        )
        m4.metric("ระยะเบี่ยงเบน ($x = |\\theta - \\theta_0|$)", f"{x_val:.2f}°")

        st.markdown("---")
        st.metric(
            label="ความหวานประเมิน (°Brix)",
            value=f"{brix_val:.2f} °Brix",
            delta=f"{'ระดับหวานมาก' if brix_val >= 15.0 else 'ระดับหวานปกติ'}",
        )

        st.markdown("---")
        st.markdown("### 📝 สรุปการวิเคราะห์เชิง AI")
        st.info(f"**เหตุผลของ AIในการจำแนก:** {reasoning}")

        st.json(
            {
                "Selected Model": auto_model,
                "Ideal Angle (theta_0)": f"{ideal_angle:.1f}°",
                "Detected Acute Angle (phi)": f"{auto_phi:.1f}°",
                "Calculated Spiral Angle (theta)": f"{auto_theta:.1f}°",
                "Deviation Value (x)": f"{x_val:.4f}",
                "Estimated Sweetness": f"{brix_val:.2f} °Brix",
            }
        )
    else:
        st.info("👈 กรุณาอัปโหลดรูปถ่ายสับปะรดที่เมนูด้านซ้ายเพื่อเริ่มต้นวิเคราะห์")
