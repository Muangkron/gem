import json
import re
import google.generativeai as genai
import streamlit as st
from PIL import Image

# -----------------------------------------------------------------------------
# 1. ตั้งค่าโครงสร้างหน้าเว็บ (Page Configuration)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ระบบวิเคราะห์มุมเกลียวสับปะรดและประเมินค่า Brix",
    page_icon="🍍",
    layout="centered",
)

st.title("🍍 ระบบวิเคราะห์เกลียวตาสับปะรดอัตโนมัติ (AI Brix Estimator)")
st.caption(
    "อัปโหลดรูปถ่ายสับปะรด เพื่อให้ Gemini AI ตรวจจับมุมเกลียวตา (θ) และคำนวณค่า Brix อัตโนมัติ"
)

# -----------------------------------------------------------------------------
# 2. จัดการ API Key จากระบบ Secrets
# -----------------------------------------------------------------------------
api_key = st.secrets.get("GEMINI_API_KEY", "")

with st.sidebar:
    st.header("⚙️ การตั้งค่าและสูตรที่ใช้")
    if not api_key:
        api_key = st.text_input("ใส่ Gemini API Key ของคุณ:", type="password")

    st.markdown("---")
    st.markdown("### 📐 สูตรคำนวณ Brix")
    st.markdown("**Model 1 (ทรงยาว / เกลียว 7/11/17):**")
    st.latex(r"x = |\theta - 155|")
    st.latex(r"\text{Brix} = -0.0196x^2 + 0.0045x + 16.757")

    st.markdown("**Model 2 (ทรงกลมป้อม / เกลียว 4/6/10):**")
    st.latex(r"x = |\theta - 136|")
    st.latex(r"\text{Brix} = 0.0082x^2 - 0.6667x + 16.362")


# -----------------------------------------------------------------------------
# 3. ฟังก์ชันคำนวณ Brix
# -----------------------------------------------------------------------------
def calculate_brix(theta, model_choice):
    if model_choice == "Model 1":
        x = abs(theta - 155)
        brix = (-0.0196 * (x**2)) + (0.0045 * x) + 16.757
    else:  # Model 2
        x = abs(theta - 136)
        brix = (0.0082 * (x**2)) - (0.6667 * x) + 16.362
    return x, brix


# -----------------------------------------------------------------------------
# 4. ส่วนการรับรูปภาพและประมวลผล
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "เลือกหรือลากรูปถ่ายสับปะรดมาวางที่นี่", type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="รูปสับปะรดที่อัปโหลด", use_container_width=True)

    if st.button("🚀 เริ่มวิเคราะห์ด้วย Gemini AI", type="primary"):
        if not api_key:
            st.error(
                "ไม่พบ API Key! กรุณากรอกใน Secrets บน Streamlit Cloud ก่อนครับ"
            )
        else:
            with st.spinner("Gemini AI กำลังวิเคราะห์แนวเกลียวตาสับปะรด..."):
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel("gemini-3.6-flash")

                    prompt = """
                    คุณคือระบบ AI วิเคราะห์เรขาคณิตเกลียวตาสับปะรดจากรูปถ่าย

                    จงวิเคราะห์ภาพสับปะรดตามขั้นตอนอย่างเคร่งครัด:

                    1. **การหาเส้นอ้างอิงและการวัดมุม:**
                       - สมมติเส้นอ้างอิงแนวนอน (Horizontal Baseline) ลากตัดผ่านกลางลูกสับปะรด
                       - ลากเส้นตามแนวเกลียวตาสับปะรดจาก "ซ้ายบน" เอียงลงไป "ขวาล่าง" (Top-Left to Bottom-Right)
                       - วัดมุมแหลม (Acute Angle) ที่เส้นเกลียวทำกับเส้นแนวนอนฝั่งขวา (สมมติได้เป็น ϕ เช่น 41° หรือ 51°)
                       - คำนวณมุมจริง θ = 180° - ϕ (เช่น 180° - 41° = 139° หรือ 180° - 51.5° = 128.5°)

                    2. **การจำแนกรูปแบบเกลียวและรูปทรงเพื่อเลือกโมเดล:**
                       - ประเมินรูปทรงและจำนวนแนวเกลียวตาสับปะรด:
                         * หากสับปะรดเป็นทรงยาว/เกลียวชัน (กลุ่มเกลียว 7/11/17 หรือ 5-8-13) -> ให้เลือก "Model 1"
                         * หากสับปะรดเป็นทรงกลมป้อม/เกลียวถี่ (กลุ่มเกลียว 4/6/10 หรือ 8-13-21) -> ให้เลือก "Model 2"

                    ตอบกลับเฉพาะข้อความ JSON รูปแบบนี้เท่านั้น (ห้ามมีคำเกริ่นหรือ Markdown Block):
                    {
                      "acute_angle_phi": 41.0,
                      "detected_angle_degrees": 139.0,
                      "spiral_pattern": "7/11/17 หรือ 4/6/10",
                      "selected_model": "Model 1",
                      "reasoning": "อธิบายสั้นๆ ว่าพบมุมแหลมกี่องศา คำนวณได้มุม θ เท่าใด และเลือกโมเดลจากทรง/เกลียวอย่างไร"
                    }
                    """

                    response = model.generate_content([prompt, image])

                    raw_text = response.text
                    json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)

                    if json_match:
                        data = json.loads(json_match.group(0))

                        phi = float(data.get("acute_angle_phi", 0))
                        theta = float(data.get("detected_angle_degrees", 0))
                        selected_model = data.get("selected_model", "Model 1")
                        pattern = data.get("spiral_pattern", "ไม่ระบุ")
                        reasoning = data.get("reasoning", "")

                        x, brix = calculate_brix(theta, selected_model)

                        st.success("วิเคราะห์สำเร็จ!")

                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("มุมแหลม (ϕ)", f"{phi:.1f}°")
                        col2.metric("มุมจริง θ (180°-ϕ)", f"{theta:.1f}°")
                        col3.metric("ค่าตัวแปร (x)", f"{x:.2f}")
                        col4.metric("ความหวานประเมิน", f"{brix:.2f} °Brix")

                        st.info(
                            f"**รูปแบบเกลียว/รูปทรง:** {pattern}\n\n"
                            f"**โมเดลที่เลือกใช้:** {selected_model}\n\n"
                            f"**ผลวิเคราะห์เพิ่มเติม:** {reasoning}"
                        )
                    else:
                        st.error(
                            "ไม่สามารถแปลงข้อมูลผลลัพธ์ได้ ลองใหม่อีกครั้งครับ"
                        )

                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาด: {str(e)}")
