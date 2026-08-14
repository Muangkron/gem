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
st.title("🍍 ระบบวัดมุมเกลียวตา & ประเมินความหวาน (°Brix)")

with st.sidebar:
    st.header("⚙️ 1. เลือกโมเดลสับปะรด")
    model_choice = st.radio(
        "ระบุโมเดล:",
        options=[
            "Model 5-8-13 (มุมอุดมคติ 155°)",
            "Model 8-13-21 (มุมอุดมคติ 136°)"
        ]
    )

    st.markdown("---")
    st.header("🛠️ 2. สไลเดอร์ปรับค่า Error")
    
    # สไลเดอร์ปรับชดเชยจุดตา (Eye Position Offset / Filter)
    eye_dist_ratio = st.slider("ระยะกรองจุดตาซ้ำ (% ภาพ):", 1.0, 5.0, 2.5, 0.5)
    eye_offset_y = st.slider("ขยับชดเชยตำแหน่งจุดตา Y (px):", -20, 20, 0, 1)

    # สไลเดอร์ปรับชดเชยมุม
    angle_offset = st.slider("ปรับชดเชยมุม Angle Offset (°):", -15.0, 15.0, 0.0, 0.1)

# -----------------------------------------------------------------------------
# 2. YOLO Model Load
# -----------------------------------------------------------------------------
@st.cache_resource
def load_yolo():
    return YOLO("best.pt")

# -----------------------------------------------------------------------------
# 3. Processing Functions
# -----------------------------------------------------------------------------
def get_filtered_eyes(image_path, model, img_w, img_h, ratio, offset_y):
    """ตรวจจับและปรับ Offset พิกัดจุดตา"""
    results = model(image_path)
    raw_points = []
    
    for r in results:
        for box in r.boxes:
            x, y, w, h = box.xywh[0].tolist()
            raw_points.append((x, y + offset_y))  # ประยุกต์ Eye Offset Y

    if not raw_points:
        return []

    # กรองจุดซ้ำ
    min_dist = (min(img_w, img_h) * ratio) / 100.0
    filtered = []
    for p in raw_points:
        if not any(math.hypot(p[0] - f[0], p[1] - f[1]) < min_dist for f in filtered):
            filtered.append(p)
            
    return filtered

def calculate_angle(centroids, img_h):
    """คำนวณมุมเกลียวด้วย Vector Regression"""
    if len(centroids) < 2:
        return None, None, None

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
    
    mean_x, mean_y = np.mean([p[0] for p in centroids]), np.mean([p[1] for p in centroids])
    intercept = mean_y - (m_px * mean_x)

    return m_px, intercept, raw_theta

def draw_hud(pil_img, centroids, slope, intercept, theta):
    """วาดพล็อตจุดตาและเส้นวัดมุมด้วย OpenCV"""
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w, _ = cv_img.shape

    # 1. พล็อตจุดตา (Target Circles)
    for cx, cy in centroids:
        pt = (int(cx), int(cy))
        cv2.circle(cv_img, pt, 6, (0, 255, 127), 2, cv2.LINE_AA)  # วงเขียว
        cv2.circle(cv_img, pt, 2, (0, 165, 255), -1, cv2.LINE_AA) # จุดส้ม

    # 2. วาดเส้นเกลียวและเส้นแนวนอน
    if slope is not None and intercept is not None:
        mean_x, mean_y = int(np.mean([p[0] for p in centroids])), int(np.mean([p[1] for p in centroids]))
        l_len = int(w * 0.35)
        
        x1, x2 = max(0, mean_x - l_len), min(w, mean_x + l_len)
        y1, y2 = int(slope * x1 + intercept), int(slope * x2 + intercept)

        cv2.line(cv_img, (x1, mean_y), (x2, mean_y), (255, 200, 0), 2, cv2.LINE_AA) # Baseline (ฟ้า)
        cv2.line(cv_img, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)            # Spiral Vector (แดง)

    return PIL.Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

def calc_brix(theta, model_name):
    """คำนวณ Brix ตามโมเดลที่เลือก"""
    if "Model 5-8-13" in model_name:
        x = abs(theta - 155.0)
        brix = (-0.0196 * (x**2)) + (0.0045 * x) + 16.757
    else:
        x = abs(theta - 136.0)
        brix = (0.0082 * (x**2)) - (0.6667 * x) + 16.362
    return brix, x

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

    # โหลด YOLO และประมวลผล
    try:
        model = load_yolo()
        eyes = get_filtered_eyes(temp_path, model, img_w, img_h, eye_dist_ratio, eye_offset_y)
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการรัน YOLO: {e}")
        eyes = []

    with col1:
        if len(eyes) >= 2:
            slope, intercept, raw_theta = calculate_angle(eyes, img_h)
            final_theta = raw_theta + angle_offset
            
            # วาดผลลัพธ์
            out_img = draw_hud(image, eyes, slope, intercept, final_theta)
            st.image(out_img, caption="พล็อตจุดตา + เส้นวัดมุมเกลียว", use_container_width=True)
        else:
            st.warning("ตรวจจับจุดตาได้ไม่เพียงพอต่อการหามุม")
            st.image(image, use_container_width=True)

    with col2:
        st.subheader("📊 ผลการวัดองศา & Brix")
        
        if len(eyes) >= 2:
            m1, m2 = st.columns(2)
            m1.metric("จำนวนตาที่พบ", f"{len(eyes)} จุด")
            m2.metric("มุมเกลียวสุทธิ (Final θ)", f"{final_theta:.2f}°", delta=f"{angle_offset:+.1f}°")

            # คำนวณความหวาน
            brix_value, diff_x = calc_brix(final_theta, model_choice)

            st.markdown("---")
            st.metric("🍬 ค่าความหวานประเมิน", f"{brix_value:.2f} °Brix")
            st.info(f"📍 ผลต่างจากมุมอุดมคติ ($x$): `{diff_x:.2f}°`")

    if os.path.exists(temp_path):
        os.remove(temp_path)
