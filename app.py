import math
import os
import cv2
import numpy as np
import PIL.Image
import streamlit as st
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# 1. Config & Sidebar Controls
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Pineapple Eye & Brix Analyzer", page_icon="🍍", layout="wide")
st.title("🍍 ระบบพล็อตจุดตา หมุนเส้นวัดมุมเกลียว & คำนวณความหวาน (°Brix)")

with st.sidebar:
    st.header("⚙️ 1. เลือกโมเดลสับปะรด")
    model_choice = st.radio(
        "ระบุประเภทโมเดล:",
        options=[
            "Model 5-8-13 (มุมอุดมคติ 155°)",
            "Model 8-13-21 (มุมอุดมคติ 136°)"
        ]
    )

    st.markdown("---")
    st.header("🛠️ 2. ปรับแต่งองศาและจุดตา")
    
    # สไลเดอร์กรองระยะห่างจุดซ้ำ
    eye_dist_ratio = st.slider("ระยะกรองจุดตาซ้ำ (% ของภาพ):", 1.0, 5.0, 2.5, 0.5)

    # สไลเดอร์หมุนเส้นชดเชยมุมเอียง (เมื่อปรับ เส้นสีแดงบนรูปจะหมุนตามทันที)
    angle_offset = st.slider(
        "🔄 หมุนปรับเส้นมุมเกลียว (Angle Offset °):",
        min_value=-20.0,
        max_value=20.0,
        value=0.0,
        step=0.1,
        help="ลากสไลเดอร์นี้เพื่อหมุนเส้นสีแดงให้ทาบตรงกับร่องเกลียวสับปะรดเป๊ะๆ"
    )

# -----------------------------------------------------------------------------
# 2. YOLO Model Load
# -----------------------------------------------------------------------------
@st.cache_resource
def load_yolo():
    return YOLO("best.pt")

# -----------------------------------------------------------------------------
# 3. Processing Functions
# -----------------------------------------------------------------------------
def get_filtered_eyes(image_path, model, img_w, img_h, ratio):
    """ตรวจจับและกรองจุดซ้ำ"""
    results = model(image_path)
    raw_points = []
    
    for r in results:
        for box in r.boxes:
            x, y, w, h = box.xywh[0].tolist()
            raw_points.append((x, y))

    if not raw_points:
        return []

    min_dist = (min(img_w, img_h) * ratio) / 100.0
    filtered = []
    for p in raw_points:
        if not any(math.hypot(p[0] - f[0], p[1] - f[1]) < min_dist for f in filtered):
            filtered.append(p)
            
    return filtered

def calculate_base_angle(centroids, img_h):
    """คำนวณมุมเกลียวตั้งต้น (Raw Theta) จาก Vector Regression"""
    if len(centroids) < 2:
        return None

    min_d, max_d = img_h * 0.05, img_h * 0.30
    slopes = []

    n = len(centroids)
    for i in range(n):
        for j in range(n):
            if i != j:
                p1, p2 = centroids[i], centroids[j]
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                if dx > 10 and dy > 10:
                    dist = math.hypot(dx, dy)
                    if min_d <= dist <= max_d:
                        m = dy / dx
                        if 0.35 <= m <= 2.5:
                            slopes.append(m)

    m_px = float(np.median(slopes)) if slopes else abs(float(np.polyfit([p[0] for p in centroids], [p[1] for p in centroids], 1)[0]))
    phi = math.degrees(math.atan(m_px))
    raw_theta = 180.0 - phi
    
    return raw_theta

def draw_hud_with_rotated_line(pil_img, centroids, final_theta):
    """วาดพล็อตจุดตา และคำนวณหมุนเส้นเวกเตอร์สีแดงตาม final_theta"""
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w, _ = cv_img.shape

    # 1. พล็อตจุดตา (Target Circles)
    for cx, cy in centroids:
        pt = (int(cx), int(cy))
        cv2.circle(cv_img, pt, 6, (0, 255, 127), 2, cv2.LINE_AA)  # วงกลมเขียว
        cv2.circle(cv_img, pt, 2, (0, 165, 255), -1, cv2.LINE_AA) # จุดส้มกลางตา

    if final_theta is not None and len(centroids) >= 2:
        # คำนวณจุดศูนย์กลางของกลุ่มตา (Center of Mass)
        mean_x = float(np.mean([p[0] for p in centroids]))
        mean_y = float(np.mean([p[1] for p in centroids]))

        # แปลง final_theta กลับมาเป็นความชัน m เพื่อวาดเส้นหมุนจริง
        phi_deg = 180.0 - final_theta
        m_pixel = math.tan(math.radians(phi_deg))
        intercept = mean_y - (m_pixel * mean_x)

        l_len = int(w * 0.38)
        x1, x2 = max(0, int(mean_x - l_len)), min(w, int(mean_x + l_len))
        y1, y2 = int(m_pixel * x1 + intercept), int(m_pixel * x2 + intercept)

        # 2. วาดเส้นอ้างอิงแนวนอน (Baseline - ฟ้า)
        cv2.line(cv_img, (x1, int(mean_y)), (x2, int(mean_y)), (255, 200, 0), 2, cv2.LINE_AA)

        # 3. วาดเส้นเกลียวสับปะรดที่หมุนตาม Slider (Spiral Vector - แดง)
        cv2.line(cv_img, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)

    return PIL.Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

def calc_brix(theta, model_name):
    """คำนวณ Brix ตามโมเดลที่เลือก"""
    if "Model 5-8-13" in model_name:
        ideal_deg = 155.0
        x = abs(theta - ideal_deg)
        brix = (-0.0196 * (x**2)) + (0.0045 * x) + 16.757
    else:
        ideal_deg = 136.0
        x = abs(theta - ideal_deg)
        brix = (0.0082 * (x**2)) - (0.6667 * x) + 16.362
    return brix, x, ideal_deg

# -----------------------------------------------------------------------------
# 4. Main Layout
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader("อัปโหลดรูปภาพสับปะรด", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1.2, 1])

    image = PIL.Image.open(uploaded_file).convert("RGB")
    temp_path = "temp_img.jpg"
    image.save(temp_path)
    img_w, img_h = image.size

    # โหลด YOLO และประมวลผลหาพิกัดตา
    try:
        model = load_yolo()
        eyes = get_filtered_eyes(temp_path, model, img_w, img_h, eye_dist_ratio)
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการรัน YOLO: {e}")
        eyes = []

    with col1:
        if len(eyes) >= 2:
            # คำนวณมุมตั้งต้น + รวมค่าหมุนจาก Slider
            raw_theta = calculate_base_angle(eyes, img_h)
            final_theta = raw_theta + angle_offset
            
            # วาดรูปภาพพร้อมเส้นสีแดงที่หมุนตาม final_theta
            out_img = draw_hud_with_rotated_line(image, eyes, final_theta)
            st.image(out_img, caption="ภาพแสดงจุดตาและเส้นเกลียว (ลาก Slider ด้านข้างเพื่อหมุนเส้นแดง)", use_container_width=True)
        else:
            st.warning("⚠️ ตรวจจับจุดตาได้น้อยกว่า 2 จุด ไม่สามารถสร้างเส้นวัดมุมได้")
            st.image(image, use_container_width=True)

    with col2:
        st.subheader("📊 ผลการวัดองศา & Brix")
        
        if len(eyes) >= 2:
            m1, m2 = st.columns(2)
            m1.metric("จำนวนตาที่พบ", f"{len(eyes)} จุด")
            m2.metric("มุมเกลียวสุทธิ (Final θ)", f"{final_theta:.2f}°", delta=f"{angle_offset:+.1f}°")

            # คำนวณ Brix
            brix_value, diff_x, ideal_angle = calc_brix(final_theta, model_choice)

            st.markdown("---")
            st.metric("🍬 ค่าความหวานประเมิน", f"{brix_value:.2f} °Brix")
            
            st.info(f"""
            - **มุมอุดมคติของโมเดล:** `{ideal_angle:.1f}°`
            - **ผลต่างองศา ($x = |\\theta - \\text{{ideal}}|$):** `{diff_x:.2f}°`
            """)

    if os.path.exists(temp_path):
        os.remove(temp_path)
