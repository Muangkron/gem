import math
import os
import cv2
from google import genai
import numpy as np
import PIL.Image
import PIL.ImageDraw
import streamlit as st
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# 1. Page Config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="🍍 Hybrid AI Pineapple Brix System",
    page_icon="🍍",
    layout="wide",
)

st.title("🍍 ระบบประเมินความหวานสับปะรด (Local YOLO + Math + Gemini)")
st.caption(
    "ระบบผสมผสาน: Local YOLO Detection (.pt model) ➔ Least Squares Angle"
    " Calculation ➔ Gemini Vision Reasoning"
)

# -----------------------------------------------------------------------------
# 2. Sidebar Setup & API Key Input (เพิ่มช่องกรอก API Key ตรงนี้)
# -----------------------------------------------------------------------------
api_key_secret = st.secrets.get("GEMINI_API_KEY", "")

with st.sidebar:
    st.header("🔑 การตั้งค่า API Key")
    user_gemini_key = st.text_input(
        "กรอก Gemini API Key ของคุณ:",
        value=api_key_secret,
        type="password",
        help="นำ API Key จาก Google AI Studio มาวางตรงนี้เพื่อเริ่มใช้งาน",
    )
    # เลือกรหัสจากช่องกรอกก่อน หากไม่มีค่อยดึงจาก Secrets
    GEMINI_KEY = user_gemini_key if user_gemini_key else api_key_secret

    st.markdown("---")
    st.header("📌 เกณฑ์แบบจำลองสับปะรด")
    st.markdown("""
    **Model 5-8-13:**
    - ตาขนาดใหญ่ ร่องตาห่าง
    - มุมเกลียวอุดมคติ $\\theta_0 = 155^\\circ$
    
    **Model 8-13-21:**
    - ตาขนาดเล็ก ถี่ อัดแน่น
    - มุมเกลียวอุดมคติ $\\theta_0 = 136^\\circ$
    """)


# -----------------------------------------------------------------------------
# 3. Load Local YOLO Model
# -----------------------------------------------------------------------------
@st.cache_resource
def load_yolo_model(model_path="best.pt"):
    """โหลดโมเดล YOLO จากไฟล์ใน Repo"""
    return YOLO(model_path)


# -----------------------------------------------------------------------------
# 4. Core Helper Functions
# -----------------------------------------------------------------------------
def detect_eyes_local_yolo(image_path, model):
    """ใช้โมเดล YOLO ในเครื่องตรวจจับพิกัดตา"""
    results = model(image_path)
    centroids = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x_center, y_center, w, h = box.xywh[0].tolist()
            centroids.append((x_center, y_center))

    return centroids


def calculate_regression_angle(centroids):
    """คำนวณเส้นถดถอยเชิงเส้นผ่านจุดศูนย์กลางตา หาค่ามุม phi และ theta"""
    if len(centroids) < 2:
        return None, None, None, None

    x_coords = np.array([p[0] for p in centroids], dtype=np.float64)
    y_coords = np.array([p[1] for p in centroids], dtype=np.float64)

    slope, intercept = np.polyfit(x_coords, y_coords, 1)
    abs_slope = abs(float(slope))
    phi_deg = math.degrees(math.atan(abs_slope))
    theta_deg = 180.0 - phi_deg

    return slope, intercept, phi_deg, theta_deg


def draw_visual_overlay(pil_img, centroids, slope, intercept):
    """วาดจุดตาและเส้นถดถอยทับลงบนภาพ"""
    img_copy = pil_img.copy()
    draw = PIL.ImageDraw.Draw(img_copy)
    w, h = img_copy.size

    # 1. วาดจุดตา (Green Circles)
    for cx, cy in centroids:
        draw.ellipse(
            [cx - 6, cy - 6, cx + 6, cy + 6], fill="#00FF66", outline="#FFFFFF"
        )

    # 2. วาดเส้นถดถอย (Red Regression Line)
    if slope is not None and intercept is not None:
        mean_x = int(np.mean([p[0] for p in centroids]))
        line_len = w // 3
        x1 = mean_x - line_len
        y1 = int(slope * x1 + intercept)
        x2 = mean_x + line_len
        y2 = int(slope * x2 + intercept)
        draw.line([(x1, y1), (x2, y2)], fill="#FF3D00", width=4)

    return img_copy


def analyze_with_gemini(pil_img, theta_val, api_key):
    """ส่งภาพ + มุมเกลียว ให้ Gemini ประเมินโมเดลและคำนวณ Brix"""
    client = genai.Client(api_key=api_key)

    prompt = f"""
    คุณเป็นระบบวิเคราะห์ทางชีววิทยาและพฤกษศาสตร์สับปะรด
    
    ข้อมูลอินพุตจากระบบวัดมุมพิกัดจริง:
    - มุมเกลียวสับปะรดที่คำนวณได้ (theta): {theta_val:.2f} องศา
    
    หน้าที่ของคุณ:
    1. วิเคราะห์รูปถ่ายสับปะรดเพื่อดูความหนาแน่น ขนาดตา และระยะห่างของร่องตา
    2. ตัดสินใจเลือกแบบจำลองสับปะรดที่ถูกต้องระหว่าง:
       - 'Model 5-8-13' (ตาใหญ่ ร่องตาห่าง มุมอุดมคติ 155 องศา)
       - 'Model 8-13-21' (ตาเล็ก ร่องตาถี่อัดแน่น มุมอุดมคติ 136 องศา)
    3. คำนวณค่าความหวาน (°Brix) ตามสมการ:
       - หากเลือก Model 5-8-13: ให้คำนวณ x = |theta - 155| และ Brix = (-0.0196 * x^2) + (0.0045 * x) + 16.757
       - หากเลือก Model 8-13-21: ให้คำนวณ x = |theta - 136| และ Brix = (0.0082 * x^2) - (0.6667 * x) + 16.362
    
    โปรดระบุผลการวิเคราะห์สั้นๆ ชัดเจน สรุปว่าเลือกโมเดลใด พร้อมแสดงขั้นตอนคำนวณค่า °Brix ที่ได้
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=[pil_img, prompt]
    )
    return response.text


# -----------------------------------------------------------------------------
# 5. Main UI Layout
# -----------------------------------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. อัปโหลดรูปภาพ")
    uploaded_file = st.file_uploader(
        "เลือกรูปภาพสับปะรด", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        temp_path = "temp_upload.jpg"
        image = PIL.Image.open(uploaded_file).convert("RGB")
        image.save(temp_path)

        st.image(image, caption="รูปภาพต้นฉบับ", use_container_width=True)

with col2:
    st.subheader("2. ผลการวิเคราะห์")

    if uploaded_file is not None:
        # เช็กว่ากรอก API Key หรือยัง
        if not GEMINI_KEY:
            st.warning("⚠️ กรุณากรอก Gemini API Key ในแถบด้านซ้ายก่อนเริ่มประมวลผล")
        elif st.button("🚀 เริ่มประมวลผลระบบ Hybrid AI", type="primary"):
            with st.spinner("กำลังโหลดโมเดลและตรวจจับพิกัดตา..."):
                try:
                    yolo_model = load_yolo_model("best.pt")
                    centroids = detect_eyes_local_yolo(temp_path, yolo_model)
                except Exception as e:
                    st.error(
                        f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}\nโปรดตรวจสอบว่าอัปโหลดไฟล์"
                        " best.pt ขึ้น GitHub แล้วหรือยัง"
                    )
                    centroids = []

            if len(centroids) < 2:
                st.warning(
                    "⚠️ ตรวจจับตาได้น้อยกว่า 2 จุด ไม่สามารถคำนวณเส้นถดถอยได้"
                )
            else:
                st.success(f"🟢 โมเดลตรวจจับจุดตาได้ทั้งหมด {len(centroids)} จุด")

                # คำนวณทางคณิตศาสตร์
                slope, intercept, phi, theta = calculate_regression_angle(
                    centroids
                )

                # แสดงภาพ Overlay
                overlay_img = draw_visual_overlay(
                    image, centroids, slope, intercept
                )
                st.image(
                    overlay_img,
                    caption=(
                        f"จุดตา (เขียว) | เส้นถดถอย (แดง) มุมเกลียว θ ="
                        f" {theta:.2f}°"
                    ),
                    use_container_width=True,
                )

                # ส่งต่อให้ Gemini ประเมินผล
                with st.spinner(
                    "กำลังส่งข้อมูลให้ Gemini วิเคราะห์ประเมินค่า °Brix..."
                ):
                    try:
                        gemini_result = analyze_with_gemini(
                            image, theta, GEMINI_KEY
                        )
                        st.markdown("### 🤖 ผลการวิเคราะห์จาก Gemini")
                        st.write(gemini_result)
                    except Exception as e:
                        st.error(
                            f"เกิดข้อผิดพลาดในการเรียก Gemini API: {e}\nโปรดตรวจสอบว่า API Key ถูกต้องหรือไม่"
                        )

            # ลบไฟล์ชั่วคราว
            if os.path.exists(temp_path):
                os.remove(temp_path)
