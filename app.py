import math
import os
import cv2
import numpy as np
import PIL.Image
import streamlit as st
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# 1. Config & Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Pineapple Eye & Brix Analyzer", page_icon="🍍", layout="wide")
st.title("🍍 ระบบวัดมุมเกลียวสับปะรดแบบ Interactive Line Rotation & Brix Calculator")

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
    """ตรวจจับและกรองจุดตาซ้ำ"""
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

def calculate_auto_angle(centroids, img_h):
    """คำนวณมุม Auto ตั้งต้นจาก Vector Regression"""
    if len(centroids) < 2:
        return 145.0  # Default fallback angle

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
    return 180.0 - phi

def draw_hud_with_rotatable_line(pil_img, centroids, target_theta):
    """วาดพล็อตจุดตา และวาดเส้นสีแดงที่หมุนตามมุม target_theta ตรงๆ"""
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w, _ = cv_img.shape

    # 1. พล็อตจุดตา (Eye Target Circles)
    for cx, cy in centroids:
        pt = (int(cx), int(cy))
        cv2.circle(cv_img, pt, 6, (0, 255, 127), 2, cv2.LINE_AA)  # วงกลมเขียว
        cv2.circle(cv_img, pt, 2, (0, 165, 255), -1, cv2.LINE_AA) # จุดกลางส้ม

    if len(centroids) >= 2:
        # จุดหมุนกลาง (Center of Rotation) จากพิกัดเฉลี่ยของตา
        mean_x = float(np.mean([p[0] for p in centroids]))
        mean_y = float(np.mean([p[1] for p in centroids]))

        # แปลงมุม target_theta ให้เป็น ความชัน (Slope m) เพื่อวาดเส้น
        phi_deg = 180.0 - target_theta
        # ป้องกัน tan(90) หรือค่าใกล้เคียง
        phi_rad = math.radians(max(0.1, min(89.9, phi_deg)))
        m_pixel = math.tan(phi_rad)
        intercept = mean_y - (m_pixel * mean_x)

        l_len = int(w * 0.40)
        x1, x2 = max(0, int(mean_x - l_len)), min(w, int(mean_x + l_len))
        y1, y2 = int(m_pixel * x1 + intercept), int(m_pixel * x2 + intercept)

        # 2. วาดเส้นอ้างอิงแนวนอน (Baseline 180° - สีฟ้า)
        cv2.line(cv_img, (x1, int(mean_y)), (x2, int(mean_y)), (255, 200, 0), 2, cv2.LINE_AA)

        # 3. วาดเส้นหมุนวัดมุมตาม Slider (Measurement Line - สีแดง)
        cv2.line(cv_img, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)

        # 4. วาดส่วนโค้งแสดงมุม (Angle Arc Indicator)
        cv2.ellipse(cv_img, (int(mean_x), int(mean_y)), (40, 40), 0, 180, 180 - phi_deg, (0, 255, 255), 2, cv2.LINE_AA)

    return PIL.Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

def calc_brix(theta, model_name):
    """คำนวณ Brix จากองศาของเส้นที่หมุนอยู่"""
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
# 4. Sidebar Controls
# -----------------------------------------------------------------------------
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
    st.header("🛠️ 2. ปรับตำแหน่งจุดตา")
    eye_dist_ratio = st.slider("ระยะกรองจุดตาซ้ำ (% ของภาพ):", 1.0, 5.0, 2.5, 0.5)

# -----------------------------------------------------------------------------
# 5. Main Application UI Layout
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader("อัปโหลดรูปภาพสับปะรด", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1.2, 1])

    image = PIL.Image.open(uploaded_file).convert("RGB")
    temp_path = "temp_img.jpg"
    image.save(temp_path)
    img_w, img_h = image.size

    # Detect Eyes
    try:
        model = load_yolo()
        eyes = get_filtered_eyes(temp_path, model, img_w, img_h, eye_dist_ratio)
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการรัน YOLO: {e}")
        eyes = []

    if len(eyes) >= 2:
        # คำนวณมุม Auto ไว้เป็นค่าอ้างอิง
        auto_theta = calculate_auto_angle(eyes, img_h)

        # ใช้ Session State เก็บค่ามุมที่หมุนอยู่
        if "manual_angle" not in st.session_state:
            st.session_state.manual_angle = float(round(auto_theta, 1))

        with col2:
            st.subheader("📐 หมุนเส้นวัดมุมเกลียว (Manual Rotation)")
            
            # ปุ่มดึงค่า Auto กลับมาตั้งต้น
            if st.button("🎯 ดึงมุมจาก AI Auto เป็นค่าเริ่มต้น"):
                st.session_state.manual_angle = float(round(auto_theta, 1))
                st.rerun()

            # สไลเดอร์หมุนเส้นตรงๆ
            current_theta = st.slider(
                "ปรับหมุนองศาเส้นสีแดง (Theta °):",
                min_value=95.0,
                max_value=175.0,
                value=float(st.session_state.manual_angle),
                step=0.1,
                help="ลากเพื่อหมุนเส้นสีแดงให้ตรงกับร่องเกลียวตาบนภาพ"
            )
            st.session_state.manual_angle = current_theta

            # คำนวณ Brix จากมุมที่หมุนเส้นอยู่นี้
            brix_val, diff_x, ideal_angle = calc_brix(current_theta, model_choice)

            st.markdown("---")
            st.subheader("📊 ผลการคำนวณ Brix")
            
            m1, m2 = st.columns(2)
            m1.metric("จำนวนตาที่พบ", f"{len(eyes)} จุด")
            m2.metric("มุมของเส้นปัจจุบัน (θ)", f"{current_theta:.1f}°")

            st.metric("🍬 ค่าความหวานประเมิน", f"{brix_val:.2f} °Brix")
            st.info(f"📍 มุมอุดมคติ: `{ideal_angle:.1f}°` | ผลต่าง ($x$): `{diff_x:.2f}°`")

        with col1:
            # วาดรูปโดยให้เส้นสีแดงหมุนตามค่า current_theta ที่ปรับบน Slider เป๊ะๆ
            out_img = draw_hud_with_rotatable_line(image, eyes, current_theta)
            st.image(out_img, caption=f"เส้นสีแดงกำลังวัดมุมที่ {current_theta:.1f}° (ปรับหมุนได้ที่แผงขวามือ)", use_container_width=True)

    else:
        st.warning("⚠️ ตรวจจับจุดตาได้น้อยกว่า 2 จุด ไม่สามารถสร้างจุดหมุนวัดมุมได้")
        st.image(image, use_container_width=True)

    if os.path.exists(temp_path):
        os.remove(temp_path)
