import json
import re
import google.generativeai as genai
import streamlit as st
from PIL import Image

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="ระบบวิเคราะห์มุมเกลียวสับปะรดและประเมินค่า Brix",
    page_icon="🍍",
    layout="centered",
)

st.title("🍍 ระบบวิเคราะห์เกลียวตาสับปะรดอัตโนมัติ (AI Brix Estimator)")
st.caption(
    "อัปโหลดรูปถ่ายสับปะรด เพื่อให้ Gemini AI ตรวจจับมุมเกลียวตา (θ) และคำนวณค่า Brix อัตโนมัติ"
)

# ดึง API Key
api_key = st.secrets.get("GEMINI_API_KEY", "")

with st.sidebar:
    st.header("⚙️ การตั้งค่าระบบ")
    if not api_key:
        api_key = st.text_input("ใส่ Gemini API Key ของคุณ:", type="password")

    st.markdown("---")
    st.markdown("### 📐 สูตรโมเดลที่ใช้งาน")
    st.latex(r"\text{Model 1: } x = |\theta - 155|")
    st.caption(r"$\text{Brix} = -0.0196x^2 + 0.0045x + 16.757$")
    st.latex(r"\text{Model 2: } x = |\theta - 136|")
    st.caption(r"$\text{Brix} = 0.0082x^2 - 0.6667x + 16.362$")


def calculate_brix(theta, model_choice):
    if model_choice == "Model 1":
        x = abs(theta - 155)
        brix = (-0.0196 * (x**2)) + (0.0045 * x) + 16.757
    else:
        x = abs(theta - 136)
        brix = (0.0082 * (x**2)) - (0.6667 * x) + 16.362
    return x, brix


uploaded_file = st.file_uploader(
    "เลือกหรือลากรูปถ่ายสับปะรดมาวางที่นี่", type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="รูปสับปะรดที่อัปโหลด", use_container_width=True)

    if st.button("🚀 เริ่มวิเคราะห์ด้วย Gemini AI", type="primary"):
        if not api_key:
            st.error("กรุณาใส่ Gemini API Key ก่อนเริ่มวิเคราะห์ครับ!")
        else:
            with st.spinner("Gemini AI กำลังวิเคราะห์แนวเกลียวตาสับปะรด..."):
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel("gemini-1.5-flash")

                    prompt = """
                    คุณคือระบบผู้เชี่ยวชาญด้านการวิเคราะห์ผลไม้ 
                    จงวิเคราะห์ภาพสับปะรดนี้ แล้วหาแนวเกลียวตาสับปะรดหลัก (Eye Spiral Line) 
                    คำนวณมุม θ (องศา) ของเกลียวตาสับปะรดเทียบกับแนวระนาบขนานพื้น (Horizontal 0 degrees)

                    จากนั้นเลือกโมเดลที่เหมาะสมที่สุดระหว่าง 'Model 1' หรือ 'Model 2'
                    
                    ตอบกลับเฉพาะข้อความ JSON รูปแบบนี้เท่านั้น (ห้ามมีคำเกริ่นหรือข้อความอื่น):
                    {
                      "detected_angle_degrees": 42.5,
                      "selected_model": "Model 1",
                      "reasoning": "อธิบายสั้นๆ ว่าทำไมถึงเลือกโมเดลนี้"
                    }
                    """

                    response = model.generate_content([prompt, image])

                    raw_text = response.text
                    json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)

                    if json_match:
                        data = json.loads(json_match.group(0))

                        theta = float(data.get("detected_angle_degrees", 0))
                        selected_model = data.get("selected_model", "Model 1")
                        reasoning = data.get("reasoning", "")

                        x, brix = calculate_brix(theta, selected_model)

                        st.success("วิเคราะห์สำเร็จ!")

                        col1, col2, col3 = st.columns(3)
                        col1.metric("มุมเกลียว (θ)", f"{theta:.2f}°")
                        col2.metric("ค่าตัวแปร (x)", f"{x:.2f}")
                        col3.metric("ความหวานประเมิน", f"{brix:.2f} °Brix")

                        st.info(
                            f"**โมเดลที่เลือกใช้:** {selected_model}\n\n**เหตุผล:** {reasoning}"
                        )
                    else:
                        st.error("ไม่สามารถแปลงผลลัพธ์ได้ ลองใหม่อีกครั้งครับ")

                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาด: {str(e)}")
