import json
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
    page_title="Gemini AI + Human-in-the-Loop Vision System",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 ระบบวิเคราะห์โมเดลและวัดมุมด้วย Gemini AI (พร้อมระบบมนุษย์ปรับแก้ไข)")
st.caption(
    "ให้ Gemini AI ทำงานล่วงหน้าเป็นหลัก แล้วเปิดให้ผู้ใช้ปรับแต่งค่าความผิดพลาดได้แบบ Real-time"
)

# ==========================================================
# 2. Sidebar Settings & API Key
# ==========================================================
st.sidebar.header("🔑 ตั้งค่า Gemini API")
api_key = st.sidebar.text_input(
    "Gemini API Key",
    value=st.secrets.get("GEMINI_API_KEY", ""),
    type="password",
    help="ใส่ API Key จาก Google AI Studio",
)

model_choice = st.sidebar.selectbox(
    "เลือกเวอร์ชันโมเดล Gemini",
    ["gemini-3.6-flash", "gemini-3.6-pro"],
    index=0,
    help="2.5-pro จะมีความแม่นยำทางมิติภาพสูงกว่า แต่ Flash จะประมวลผลเร็วกว่า",
)


# ==========================================================
# 3. Gemini High-Precision Vision Analyzer
# ==========================================================
def analyze_with_gemini(pil_img, key, model_name):
    """ส่งภาพให้ Gemini วิเคราะห์ด้วย Prompt ความแม่นยำสูง และบังคับโครงสร้าง JSON"""
    if not key:
        return None

    try:
        client = genai.Client(api_key=key)

        system_instruction = (
            "You are a precise computer vision and mechanical measurement AI. "
            "Your task is to analyze images of objects/models, measure their orientation angle, "
            "classify their model category, and locate their bounding box coordinates with high spatial accuracy."
        )

        prompt = """
        Analyze this image carefully:
        1. **Model Classification**: Identify the precise model type/category of the main subject.
        2. **Angle Measurement**: Determine the primary orientation/tilt angle in degrees (from -180.0 to 180.0). Horizontal line is 0 degrees.
        3. **Bounding Box**: Provide the bounding box containing the model using normalized coordinates [ymin, xmin, ymax, xmax] scaled from 0 to 1000.
        4. **Confidence**: Provide a confidence score percentage (0-100).

        Respond STRICTLY in JSON format with this exact structure:
        {
            "model_type": "string",
            "detected_angle": float,
            "bounding_box_1000": [ymin, xmin, ymax, xmax],
            "confidence_score": float,
            "reasoning": "string explanation of classification"
        }
        """

        response = client.models.generate_content(
            model=model_name,
            contents=[pil_img, prompt],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                temperature=0.0,  # กำหนด 0.0 เพื่อลด Hallucination และให้ค่าที่นิ่งที่สุด
            ),
        )

        return json.loads(response.text)
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดจาก Gemini API: {str(e)}")
        return None


# ==========================================================
# 4. Main Application Execution
# ==========================================================
uploaded_file = st.file_uploader(
    "อัปโหลดรูปภาพเพื่อเริ่มการวิเคราะห์ (JPG, PNG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    # โหลดไฟล์ภาพ
    pil_image = Image.open(uploaded_file).convert("RGB")
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    img_h, img_w = cv_image.shape[:2]

    # ปุ่มสำหรับกดเรียก Gemini ประมวลผลใหม่
    st.markdown("---")
    if st.button("🚀 สั่ง Gemini AI วิเคราะห์ภาพ (Analyze Image)"):
        if not api_key:
            st.error("กรุณากรอก Gemini API Key ในเมนูด้านซ้ายก่อนทำรายการ")
        else:
            with st.spinner("🧠 Gemini AI กำลังวัดมุมและวิเคราะห์จำแนกโมเดล..."):
                res = analyze_with_gemini(pil_image, api_key, model_choice)

                if res:
                    # แปลง Bounding Box (0-1000) มาเป็นพิกัด Pixel จริง
                    g_box = res.get(
                        "bounding_box_1000", [100, 100, 900, 900]
                    )
                    ymin = int((g_box[0] / 1000.0) * img_h)
                    xmin = int((g_box[1] / 1000.0) * img_w)
                    ymax = int((g_box[2] / 1000.0) * img_h)
                    xmax = int((g_box[3] / 1000.0) * img_w)

                    # บันทึกค่าลงใน Session State เพื่อนำไปตั้งค่าเริ่มต้นให้สไลเดอร์มนุษย์
                    st.session_state["gemini_analyzed"] = True
                    st.session_state["ai_model_type"] = res.get(
                        "model_type", "Unknown Model"
                    )
                    st.session_state["ai_angle"] = float(
                        res.get("detected_angle", 0.0)
                    )
                    st.session_state["ai_confidence"] = float(
                        res.get("confidence_score", 0.0)
                    )
                    st.session_state["ai_reasoning"] = res.get(
                        "reasoning", ""
                    )

                    st.session_state["roi_xmin"] = max(0, xmin)
                    st.session_state["roi_ymin"] = max(0, ymin)
                    st.session_state["roi_xmax"] = min(img_w, xmax)
                    st.session_state["roi_ymax"] = min(img_h, ymax)
                    st.success("วิเคราะห์สำเร็จ! คุณสามารถปรับแต่งค่าผลลัพธ์เพิ่มเติมได้ที่แผงควบคุมด้านล่าง")

    # เช็คว่ามีข้อมูลวิเคราะห์เดิมหรือยัง
    if "gemini_analyzed" not in st.session_state:
        st.session_state["gemini_analyzed"] = False
        st.session_state["ai_model_type"] = "ยังไม่ได้วิเคราะห์"
        st.session_state["ai_angle"] = 0.0
        st.session_state["ai_confidence"] = 0.0
        st.session_state["ai_reasoning"] = "-"
        st.session_state["roi_xmin"] = int(img_w * 0.1)
        st.session_state["roi_ymin"] = int(img_h * 0.1)
        st.session_state["roi_xmax"] = int(img_w * 0.9)
        st.session_state["roi_ymax"] = int(img_h * 0.9)

    # ------------------------------------------------------
    # 5. Human-in-the-Loop Adjustment Controls (แถบควบคุมของมนุษย์)
    # ------------------------------------------------------
    st.markdown("---")
    st.subheader("🛠️ แผงควบคุมและปรับแต่งแก้ไขผลลัพธ์โดยมนุษย์ (Human Fine-Tuning)")

    ctrl_col1, ctrl_col2 = st.columns(2)

    with ctrl_col1:
        st.markdown("##### 1. ปรับแต่งประเภทโมเดลและมุมองศา")
        user_model = st.text_input(
            "ชื่อ/ประเภทโมเดล (ปรับแก้ได้)",
            value=st.session_state["ai_model_type"],
        )
        user_angle = st.slider(
            "ปรับแก้องศาการหมุน/มุมเอียง (Degrees)",
            min_value=-180.0,
            max_value=180.0,
            value=float(st.session_state["ai_angle"]),
            step=0.1,
        )

    with ctrl_col2:
        st.markdown("##### 2. ปรับแต่งกรอบ ROI (Bounding Box)")
        box_c1, box_c2 = st.columns(2)
        with box_c1:
            user_xmin = st.number_input(
                "X Min (พิกเซล)",
                min_value=0,
                max_value=img_w,
                value=int(st.session_state["roi_xmin"]),
            )
            user_ymin = st.number_input(
                "Y Min (พิกเซล)",
                min_value=0,
                max_value=img_h,
                value=int(st.session_state["roi_ymin"]),
            )
        with box_c2:
            user_xmax = st.number_input(
                "X Max (พิกเซล)",
                min_value=0,
                max_value=img_w,
                value=int(st.session_state["roi_xmax"]),
            )
            user_ymax = st.number_input(
                "Y Max (พิกเซล)",
                min_value=0,
                max_value=img_h,
                value=int(st.session_state["roi_ymax"]),
            )

    # ------------------------------------------------------
    # 6. Real-time Rendering & Auto-Crop Display
    # ------------------------------------------------------
    st.markdown("---")
    disp_col1, disp_col2 = st.columns(2)

    # การวาดกรอบตามค่าที่มนุษย์ปรับแต่งปรับแก้
    annotated_img = cv_image.copy()
    cv2.rectangle(
        annotated_img,
        (user_xmin, user_ymin),
        (user_xmax, user_ymax),
        (0, 255, 0),
        3,
    )

    # เขียนข้อความบอกมุมบนภาพ
    cv2.putText(
        annotated_img,
        f"Angle: {user_angle:.1f} deg",
        (user_xmin, max(20, user_ymin - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    # Auto-Crop ภาพ
    crop_x1 = max(0, min(user_xmin, user_xmax))
    crop_y1 = max(0, min(user_ymin, user_ymax))
    crop_x2 = min(img_w, max(user_xmin, user_xmax))
    crop_y2 = min(img_h, max(user_ymin, user_ymax))

    cropped_img = cv_image[crop_y1:crop_y2, crop_x1:crop_x2]

    with disp_col1:
        st.subheader("1. ภาพอ้างอิง + กรอบ ROI ปัจจุบัน")
        st.image(
            cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
            use_container_width=True,
        )

    with disp_col2:
        st.subheader("2. ภาพ Auto-Crop ล่าสุด")
        if cropped_img.size > 0:
            st.image(
                cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )
        else:
            st.warning("กรอบ ROI ไม่ถูกต้อง ไม่สามารถตัดภาพได้")

    # ------------------------------------------------------
    # 7. Final Summary Readout
    # ------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 ผลสรุปการประมวลผล (Final Verified Results)")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(label="มุมสรุปหลังปรับแก้ไข", value=f"{user_angle:.1f}°")
    with m2:
        st.metric(label="ประเภทโมเดลสรุป", value=user_model)
    with m3:
        st.metric(
            label="ความน่าเชื่อถือจาก AI",
            value=f"{st.session_state['ai_confidence']}%",
        )

    if st.session_state["ai_reasoning"] != "-":
        st.info(
            f"💡 **เหตุผลการวิเคราะห์เบื้องต้นจาก Gemini:** {st.session_state['ai_reasoning']}"
        )
else:
    st.info("💡 กรุณาอัปโหลดรูปภาพเพื่อเริ่มต้นใช้งานระบบ")
