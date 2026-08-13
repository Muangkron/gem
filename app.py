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
# 1. Page Config & Secrets Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="🍍 Hybrid AI Pineapple Brix System",
    page_icon="🍍",
    layout="wide",
)

st.title("🍍 ระบบประเมินความหวานสับปะรด (Local YOLO + Math + Gemini)")
st.caption(
    "ระบบผสมผสาน: Local YOLO Detection (.pt model) ➔ Least Squares Angle Calculation ➔ Gemini Vision Reasoning"
)

# ดึงเฉพาะ Gemini Key (ไม่ต้องใช้ Roboflow Key แล้ว)
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", "")


# -----------------------------------------------------------------------------
# 2. Load Local YOLO Model
# -----------------------------------------------------------------------------
@st.cache_resource
def load_yolo_model(model_path="best.pt"):
    """โหลดโมเดล YOLO จากไฟล์ใน Repo"""
    return YOLO(model_path)


# -----------------------------------------------------------------------------
# 3. Core Helper Functions
# -----------------------------------------------------------------------------
def detect_eyes_local_yolo(image_path, model):
    """ใช้โมเดล YOLO ในเครื่องตรวจจับพิกัดตา"""
    results = model(image_path)
    centroids = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # ดึงพิกัดจุดศูนย์กลาง x_center, y_center ของ Bounding Box
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


def analyze_with_gemini(pil_img, theta_val, weight_kg):
    """ส่งภาพ + มุม + น้ำหนัก ไปให้ Gemini ประเมินโมเดลและคำนวณ Brix"""
    client = genai.Client(api_key=GEMINI_KEY)

    prompt = f"""
    คุณเป็นระบบวิเคราะห์ทางชีววิทยาและพฤกษศาสตร์สับปะรด
    
    ข้อมูลอินพุตจากระบบวัดมุมพิกัดจริง:
    - มุมเกลียวสับปะรดที่คำนวณได้ (theta): {theta_val:.2f} องศา
    - น้ำหนักผลประมาณ: {weight_kg:.2f} กิโลกรัม
    
    หน้าที่ของคุณ:
    1. ตรวจสอบภาพถ่ายสับปะรดเพื่อดูความหนาแน่นของตาและลวดลายผล
    2. ตัดสินใจเลือกโมเดลที่ถูกต้องระหว่าง:
       - 'Model 5-8-13' (สำหรับผลเล็ก-กลาง ตาใหญ่ห่าง มุมอุดมคติ 155 องศา)
       - 'Model 8-13-21' (สำหรับผลกลาง-ใหญ่ ตาเล็กถี่อัดแน่น มุมอุดมคติ 136 องศา)
    3. คำนวณค่าความหวาน (°Brix) ตามสมการ:
       - หากเลือก Model 5-8-13: ให้คำนวณ x = |theta - 155| และ Brix = (-0.0196 * x^2) + (0.0045 * x) + 16.757
       - หากเลือก Model 8-13-21: ให้คำนวณ x = |theta - 136| และ Brix = (0.0082 * x^2) - (0.6667 * x) + 16.362
    
    โปรดสรุปผลการวิเคราะห์สั้นๆ เป็นระเบียบ ชัดเจน พร้อมระบุเหตุผลในการเลือกโมเดล และสรุปค่า °Brix ที่ได้
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=[pil_img, prompt]
    )
    return response.text


# -----------------------------------------------------------------------------
# 4. Streamlit UI Layout
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ ตั้งค่าข้อมูลกายภาพ")
    weight_input = st.number_input(
        "น้ำหนักผล (กิโลกรัม):",
        min_value=0.3,
        max_value=4.0,
        value=1.1,
        step=0.1,
    )
    st.info("💡 น้ำหนัก < 1.2 kg มักเป็น 5-8-13 | น้ำหนัก >= 1.2 kg มักเป็น 8-13-21")

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
    if uploaded_file is not None and st.button(
        "🚀 เริ่มประมวลผลระบบ Hybrid AI", type="primary"
    ):
        with st.spinner("กำลังโหลดโมเดลและตรวจจับพิกัดตา..."):
            try:
                yolo_model = load_yolo_model("best.pt")
                centroids = detect_eyes_local_yolo(temp_path, yolo_model)
            except Exception as e:
                st.error(
                    f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}\nโปรดตรวจสอบว่ามีไฟล์ best.pt ใน GitHub แล้วหรือยัง"
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
                caption=f"จุดตา (เขียว) | เส้นถดถอย (แดง) มุมเกลียว θ = {theta:.2f}°",
                use_container_width=True,
            )

            # ส่งต่อให้ Gemini ประเมินผล
            with st.spinner("กำลังส่งข้อมูลให้ Gemini วิเคราะห์ประเมินค่า °Brix..."):
                try:
                    gemini_result = analyze_with_gemini(
                        image, theta, weight_input
                    )
                    st.markdown("### 🤖 ผลการวิเคราะห์จาก Gemini")
                    st.write(gemini_result)
                except Exception as e:
                    st.error(
                        f"เกิดข้อผิดพลาดในการเรียก Gemini API: {e}\nกรุณาตรวจสอบ GEMINI_API_KEY ใน Streamlit Secrets"
                    )

        # ลบไฟล์ชั่วคราวหลังใช้งานเสร็จ
        if os.path.exists(temp_path):
            os.remove(temp_path)
