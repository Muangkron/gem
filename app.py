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

# แถบข้าง (Sidebar) แสดงรายละเอียดสูตรคำนวณ
with st.sidebar:
    st.header("⚙️ การตั้งค่าและสูตรที่ใช้")
    if not api_key:
        api_key = st.text_input("ใส่ Gemini API Key ของคุณ:", type="password")

    st.markdown("---")
    st.markdown("### 📐 สูตรคำนวณ Brix")
    st.markdown("**Model 1 (เกลียวแบบ 5-8-13):**")
    st.latex(r"x = |\theta - 155|")
    st.latex(r"\text{Brix} = -0.0196x^2 + 0.0045x + 16.757")

    st.markdown("**Model 2 (เกลียวแบบ 8-13-21):**")
    st.latex(r"x = |\theta - 136|")
    st.latex(r"\text{Brix} = 0.0082x^2 - 0.6667x + 16.362")


# -----------------------------------------------------------------------------
# 3. ฟังก์ชันคำนวณ Brix ด้วย Python (การันตีความแม่นยำของคณิตศาสตร์)
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
# 4. ส่วนการทำงานหลัก (Upload, Vision AI & Results)
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "เลือกหรือลากรูปถ่ายสับปะรดมาวางที่นี่", type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    # อ่านไฟล์รูปภาพ
    image = Image.open(uploaded_file)
    st.image(
        image, caption="รูปสับปะรดที่อัปโหลด", use_container_width=True
    )

    if st.button("🚀 เริ่มวิเคราะห์ด้วย Gemini AI", type="primary"):
        if not api_key:
            st.error(
                "ไม่พบ API Key! กรุณากรอกใน Secrets บน Streamlit Cloud ก่อนครับ"
            )
        else:
            with st.spinner("Gemini AI กำลังวิเคราะห์แนวเกลียวตาสับปะรด..."):
                try:
                    # ตั้งค่าโมเดล Gemini
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel("gemini-3.6-flash")

                    # ชุดคำสั่งระบุเกณฑ์การวัดมุมและเลือกโมเดล
                    prompt = """
                    คุณคือ AI ผู้เชี่ยวชาญด้านเรขาคณิตและโครงสร้างผลไม้ (Pineapple Parastichy & Spiral Analysis)
                    
                    จงวิเคราะห์ภาพสับปะรดนี้อย่างละเอียดตามขั้นตอนต่อไปนี้:

                    1. **ระบบการวัดมุม (Angle Measurement):**
                       - ให้หาแนวเส้นเกลียวตาสับปะรดหลัก (Eye Spiral Line)
                       - วัดมุม θ (องศา) โดยเริ่มวัดจาก "แกน +X ในแนวนอน" (Horizontal 0 degrees ชี้ไปทางขวา) แล้วหมุน "ทวนเข็มนาฬิกา" (Counter-Clockwise) ขึ้นไปจนถึงแนวเส้นเกลียวตาสับปะรด

                    2. **การเลือกโมเดลจากรูปแบบเกลียวตา (Parastichy Pattern):**
                       - สแกนนับหรือประเมินความหนาแน่นและจำนวนแนวเกลียวตาตามลำดับฟีโบนักชี (Fibonacci Series):
                         * หากสับปะรดอยู่ในกลุ่มลวดลาย **5-8-13** (แนวเกลียวชันมาตรฐาน) -> ให้เลือก **"Model 1"**
                         * หากสับปะรดอยู่ในกลุ่มลวดลาย **8-13-21** (แนวเกลียวชันมาก/ตาถี่) -> ให้เลือก **"Model 2"**

                    ตอบกลับเฉพาะข้อความ JSON รูปแบบนี้เท่านั้น (ห้ามมีคำเกริ่นหรือ Markdown Block):
                    {
                      "spiral_pattern": "5-8-13 หรือ 8-13-21",
                      "detected_angle_degrees": 45.0,
                      "selected_model": "Model 1",
                      "reasoning": "อธิบายสั้นๆ ว่าพบเกลียวรูปแบบใด และวัดมุมทวนเข็มจากแกน +X ได้กี่องศา"
                    }
                    """

                    # ส่งรูปภาพและคำสั่งไปประมวลผลที่ Gemini
                    response = model.generate_content([prompt, image])

                    # ดึงข้อมูล JSON ออกจากคำตอบ
                    raw_text = response.text
                    json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)

                    if json_match:
                        data = json.loads(json_match.group(0))

                        theta = float(data.get("detected_angle_degrees", 0))
                        selected_model = data.get("selected_model", "Model 1")
                        pattern = data.get("spiral_pattern", "ไม่ระบุ")
                        reasoning = data.get("reasoning", "")

                        # คำนวณค่า Brix ด้วยคำสั่ง Python
                        x, brix = calculate_brix(theta, selected_model)

                        st.success("วิเคราะห์สำเร็จ!")

                        # แสดงการ์ดตัวเลขสรุปผล
                        col1, col2, col3 = st.columns(3)
                        col1.metric("มุมเกลียวทวนเข็ม (θ)", f"{theta:.2f}°")
                        col2.metric("ค่าตัวแปร (x)", f"{x:.2f}")
                        col3.metric("ความหวานประเมิน", f"{brix:.2f} °Brix")

                        # รายละเอียดเพิ่มเติม
                        st.info(
                            f"**รูปแบบเกลียวตาที่พบ:** {pattern}\n\n"
                            f"**โมเดลที่เลือกใช้:** {selected_model}\n\n"
                            f"**วิเคราะห์เพิ่มเติม:** {reasoning}"
                        )
                    else:
                        st.error(
                            "ไม่สามารถแปลงข้อมูลผลลัพธ์ได้ กรุณาลองใหม่อีกครั้งครับ"
                        )

                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในการประมวลผล: {str(e)}")
