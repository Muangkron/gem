import math
import os
import cv2
import google.generativeai as genai
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

st.title("🍍 ระบบประเมินความหวานสับปะรด (Local YOLO + Spiral Math + Gemini)")
st.caption("ระบบประมวลผล: YOLO Eye Detection ➔ Duplicate Filtering ➔ Spiral Vector Matching ➔ Gemini Vision")

# -----------------------------------------------------------------------------
# 2. Sidebar Settings
# -----------------------------------------------------------------------------
api_key_secret = st.secrets.get("GEMINI_API_KEY", "")

with st.sidebar:
    st.header("🔑 ตั้งค่า API Key")
    user_gemini_key = st.text_input(
        "Gemini API Key:",
        value=api_key_secret,
        type="password",
        help="นำ API Key จาก Google AI Studio มาวางตรงนี้",
    )
    GEMINI_KEY = user_gemini_key if user_gemini_key else api_key_secret

    st.markdown("---")
    st.header("⚙️ ปรับแต่งระบบวัดเกลียว")
    dist_threshold_ratio = st.slider(
        "ระยะกรองจุดตาซ้ำ (% ของภาพ):",
        min_value=1.0,
        max_value=6.0,
        value=2.5,
        step=0.5
    )

# -----------------------------------------------------------------------------
# 3. Load Local YOLO Model
# -----------------------------------------------------------------------------
@st.cache_resource
def load_yolo_model(model_path="best.pt"):
    return YOLO(model_path)

# -----------------------------------------------------------------------------
# 4. Helper Functions
# -----------------------------------------------------------------------------
def detect_and_filter_eyes(image_path, model, img_w, img_h, ratio=2.5):
    results = model(image_path)
    raw_centroids = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x_center, y_center, w, h = box.xywh[0].tolist()
            raw_centroids.append((x_center, y_center))

    if not raw_centroids:
        return []

    min_dist_px = (min(img_w, img_h) * ratio) / 100.0
    filtered_centroids = []

    for c in raw_centroids:
        is_duplicate = False
        for f in filtered_centroids:
            dist = math.hypot(c[0] - f[0], c[1] - f[1])
            if dist < min_dist_px:
                is_duplicate = True
                break
        if not is_duplicate:
            filtered_centroids.append(c)

    return filtered_centroids

def calculate_accurate_spiral_angle(centroids, img_w, img_h):
    if len(centroids) < 2:
        return None, None, None, None

    min_neighbor_dist = img_h * 0.05
    max_neighbor_dist = img_h * 0.30
    spiral_slopes = []

    n = len(centroids)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p1, p2 = centroids[i], centroids[j]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]

            if dx > 10 and dy > 10:
                dist = math.hypot(dx, dy)
                if min_neighbor_dist <= dist <= max_neighbor_dist:
                    slope_px = dy / dx
                    if 0.35 <= slope_px <= 2.5:
                        spiral_slopes.append(slope_px)

    if spiral_slopes:
        m_pixel = float(np.median(spiral_slopes))
    else:
        x_coords = np.array([p[0] for p in centroids], dtype=np.float64)
        y_coords = np.array([p[1] for p in centroids], dtype=np.float64)
        m_pixel, _ = np.polyfit(x_coords, y_coords, 1)
        m_pixel = abs(float(m_pixel))

    phi_deg = math.degrees(math.atan(m_pixel))
    theta_deg = 180.0 - phi_deg

    mean_x = float(np.mean([p[0] for p in centroids]))
    mean_y = float(np.mean([p[1] for p in centroids]))
    intercept = mean_y - (m_pixel * mean_x)

    return m_pixel, intercept, phi_deg, theta_deg

def draw_visual_overlay(pil_img, centroids, slope, intercept, phi_deg, theta_deg):
    img_copy = pil_img.copy()
    draw = PIL.ImageDraw.Draw(img_copy)
    w, h = img_copy.size

    x_coords = [p[0] for p in centroids]
    mean_x = float(np.mean(x_coords))
    mean_y = float(np.mean([p[1] for p in centroids]))

    line_length = int(w * 0.4)
    x_min = max(0, int(mean_x - line_length))
    x_max = min(w, int(mean_x + line_length))

    circle_radius = max(5, int(min(w, h) * 0.009))
    for cx, cy in centroids:
        draw.ellipse(
            [cx - circle_radius, cy - circle_radius, cx + circle_radius, cy + circle_radius],
            fill="#00FF66", outline="#000000", width=2
        )

    if slope is not None and intercept is not None:
        draw.line([(x_min, mean_y), (x_max, mean_y)], fill="#FF0000", width=4)
        y1, y2 = slope * x_min + intercept, slope * x_max + intercept
        draw.line([(x_min, int(y1)), (x_max, int(y2))], fill="#FF0000", width=4)

        text_info = f"Spiral Angle (Theta) = {theta_deg:.1f} deg"
        draw.text((max(10, x_min), max(10, int(mean_y) - 45)), text_info, fill="#FFFF00")

    return img_copy

def analyze_with_gemini(pil_img, theta_val, api_key):
    fast_img = pil_img.copy()
    fast_img.thumbnail((800, 800))

    genai.configure(api_key=api_key)

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

    # รายชื่อโมเดลที่เรียงตามความเสถียรและเวอร์ชันล่าสุดบน Google API
    candidate_models = [
        "gemini-2.0-flash",
        "gemini-1.5-flash-002",
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro-latest",
        "models/gemini-1.5-flash"
    ]

    last_error = None
    for model_name in candidate_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content([fast_img, prompt])
            return response.text, model_name
        except Exception as e:
            last_error = e
            continue

    # หากทุกโมเดลล้มเหลว ให้โยน Error ออกมาแจ้งเตือน
    raise last_error

# -----------------------------------------------------------------------------
# 5. Main UI Layout
# -----------------------------------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. อัปโหลดรูปภาพ")
    uploaded_file = st.file_uploader("เลือกรูปภาพสับปะรด", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        temp_path = "temp_upload.jpg"
        image = PIL.Image.open(uploaded_file).convert("RGB")
        image.save(temp_path)
        img_w, img_h = image.size
        st.image(image, caption="รูปภาพต้นฉบับ", use_container_width=True)

with col2:
    st.subheader("2. ผลการวิเคราะห์")

    if uploaded_file is not None:
        if not GEMINI_KEY:
            st.warning("⚠️ กรุณากรอก Gemini API Key ในแถบด้านซ้ายก่อนเริ่มประมวลผล")
        elif st.button("🚀 เริ่มประมวลผลระบบ Hybrid AI", type="primary"):
            with st.spinner("กำลังตรวจจับพิกัดตา และคำนวณมุมเกลียว..."):
                try:
                    yolo_model = load_yolo_model("best.pt")
                    centroids = detect_and_filter_eyes(temp_path, yolo_model, img_w, img_h, dist_threshold_ratio)
                except Exception as e:
                    st.error("เกิดข้อผิดพลาดในการรันโมเดล YOLO:")
                    st.exception(e)
                    centroids = []

            if len(centroids) < 2:
                st.warning("⚠️ ตรวจจับตาได้น้อยกว่า 2 จุด ไม่สามารถคำนวณเส้นถดถอยได้")
            else:
                slope, intercept, phi, theta = calculate_accurate_spiral_angle(centroids, img_w, img_h)

                st.success(f"🟢 ตรวจจับตาได้ {len(centroids)} จุด | คำนวณมุมเกลียว θ = {theta:.1f}°")

                overlay_img = draw_visual_overlay(image, centroids, slope, intercept, phi, theta)
                st.image(overlay_img, caption=f"มุมเกลียวสับปะรด θ = {theta:.1f}°", use_container_width=True)

                with st.spinner("กำลังส่งข้อมูลให้ Gemini วิเคราะห์..."):
                    try:
                        gemini_result, used_model = analyze_with_gemini(image, theta, GEMINI_KEY)
                        st.caption(f"✨ ประมวลผลสำเร็จผ่านโมเดล: `{used_model}`")
                        st.markdown("### 🤖 ผลการวิเคราะห์จาก Gemini")
                        st.write(gemini_result)
                    except Exception as e:
                        st.error("เกิดข้อผิดพลาดในการเรียก Gemini API:")
                        st.exception(e)

            if os.path.exists(temp_path):
                os.remove(temp_path)
