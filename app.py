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
st.set_page_config(page_title="Pineapple Straightener & Brix Analyzer", page_icon="🍍", layout="wide")
st.title("🍍 ระบบดัดสับปะรดตั้งตรง + หมุนเส้นวัดมุมเกลียวแบบ Smooth & Brix Calculator")

# -----------------------------------------------------------------------------
# 2. YOLO Model Load
# -----------------------------------------------------------------------------
@st.cache_resource
def load_yolo():
    return YOLO("best.pt")

# -----------------------------------------------------------------------------
# 3. Image Processing & Alignment Functions
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

def straighten_pineapple_image(cv_img, centroids):
    """
    คำนวณมุมเอียงหลักของผลสับปะรด และทำการหมุนรูปภาพ (Image Rotation)
    เพื่อให้ผลสับปะรดตั้งตรงเป๊ะ พร้อมปรับพิกัดจุดตาตามการหมุน
    """
    if len(centroids) < 2:
        return cv_img, centroids

    h, w = cv_img.shape[:2]
    mean_x = float(np.mean([p[0] for p in centroids]))
    mean_y = float(np.mean([p[1] for p in centroids]))

    # หาแกนความเอียงหลักของกลุ่มตาด้วย Principal Component Analysis (PCA)
    pts = np.array(centroids, dtype=np.float32)
    mean, eigenvectors = cv2.pcaCompute(pts, mean=np.empty((0)))
    
    # คำนวณมุมเอียงของแกนหลัก (Tilt Angle)
    angle_rad = math.atan2(eigenvectors[0, 1], eigenvectors[0, 0])
    tilt_deg = math.degrees(angle_rad)

    # ปรับแกนให้อยู่ในแนวตั้ง (Upright Position ~ -90° หรือ 90°)
    if tilt_deg > 45:
        correction_deg = tilt_deg - 90
    elif tilt_deg < -45:
        correction_deg = tilt_deg + 90
    else:
        correction_deg = tilt_deg

    # หมุนรูปภาพรอบจุดศูนย์กลางกลุ่มตา
    rot_matrix = cv2.getRotationMatrix2D((mean_x, mean_y), correction_deg, 1.0)
    straightened_img = cv2.warpAffine(
        cv_img, rot_matrix, (w, h), 
        flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(255, 255, 255)
    )

    # Transform พิกัดจุดตาไปยังตำแหน่งใหม่ในภาพที่หมุนแล้ว
    ones = np.ones(shape=(len(centroids), 1))
    points_ones = np.hstack([pts, ones])
    transformed_pts = rot_matrix.dot(points_ones.T).T
    new_centroids = [(p[0], p[1]) for p in transformed_pts]

    return straightened_img, new_centroids

def calculate_auto_angle(centroids, img_h):
    """คำนวณมุม Auto ตั้งต้นจาก Vector Regression บนภาพที่ตั้งตรงแล้ว"""
    if len(centroids) < 2:
        return 145.0

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

def draw_hud_smooth(cv_img, centroids, target_theta):
    """วาดพล็อตจุดตา และวาดเส้นหมุนวัดมุมที่สมูท คมชัดระดับ Sub-pixel"""
    img_out = cv_img.copy()
    h, w, _ = img_out.shape

    # 1. พล็อตจุดตาบนภาพตั้งตรง
    for cx, cy in centroids:
        pt = (int(cx), int(cy))
        cv2.circle(img_out, pt, 6, (0, 255, 127), 2, cv2.LINE_AA)  # วงเขียว
        cv2.circle(img_out, pt, 2, (0, 165, 255), -1, cv2.LINE_AA) # จุดส้มกลางตา

    if len(centroids) >= 2:
        mean_x = float(np.mean([p[0] for p in centroids]))
        mean_y = float(np.mean([p[1] for p in centroids]))

        # แปลงมุมองศาเป้าหมายเป็น Slope m
        phi_deg = 180.0 - target_theta
        phi_rad = math.radians(max(0.01, min(89.99, phi_deg)))
        m_pixel = math.tan(phi_rad)
        intercept = mean_y - (m_pixel * mean_x)

        l_len = int(w * 0.42)
        x1, x2 = max(0, int(mean_x - l_len)), min(w, int(mean_x + l_len))
        y1, y2 = int(m_pixel * x1 + intercept), int(m_pixel * x2 + intercept)

        # 2. วาดเส้นอ้างอิงแนวนอนขนานพื้น (Baseline 180° - สีฟ้า)
        cv2.line(img_out, (x1, int(mean_y)), (x2, int(mean_y)), (255, 200, 0), 2, cv2.LINE_AA)

        # 3. วาดเส้นหมุนวัดมุมตาม Slider (Measurement Line - สีแดง)
        cv2.line(img_out, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)

        # 4. วาดส่วนโค้งแสดงองศา (Angle Arc Indicator)
        cv2.ellipse(img_out, (int(mean_x), int(mean_y)), (45, 45), 0, 180, 180 - phi_deg, (0, 255, 255), 2, cv2.LINE_AA)

    return PIL.Image.fromarray(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))

def calc_brix(theta, model_name):
    """คำนวณ Brix จากมุมเส้นที่หมุนอยู่"""
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
    st.header("🛠️ 2. ปรับแต่งระบบ")
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

    # โหลด YOLO ตรวจจับจุดตา
    try:
        model = load_yolo()
        raw_eyes = get_filtered_eyes(temp_path, model, img_w, img_h, eye_dist_ratio)
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการรัน YOLO: {e}")
        raw_eyes = []

    if len(raw_eyes) >= 2:
        # 1. ดัดภาพสับปะรดให้ตั้งตรงอัตโนมัติ (Straighten Pineapple)
        cv_img_orig = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        straight_cv_img, straight_eyes = straighten_pineapple_image(cv_img_orig, raw_eyes)

        # 2. คำนวณมุม Auto บนภาพที่ตั้งตรงแล้ว
        auto_theta = calculate_auto_angle(straight_eyes, img_h)

        if "manual_angle" not in st.session_state:
            st.session_state.manual_angle = float(round(auto_theta, 2))

        with col2:
            st.subheader("📐 หมุนเส้นวัดมุมเกลียว (Ultra-Smooth Slider)")
            
            if st.button("🎯 ดึงมุม Auto เป็นค่าเริ่มต้น"):
                st.session_state.manual_angle = float(round(auto_theta, 2))
                st.rerun()

            # สไลเดอร์ระดับ ละเอียด 0.01 องศา เพื่อความสมูทสูงสุด
            current_theta = st.slider(
                "ปรับหมุนองศาเส้นสีแดง (Theta °):",
                min_value=90.00,
                max_value=180.00,
                value=float(st.session_state.manual_angle),
                step=0.01,
                format="%.2f°",
                help="ลากสไลเดอร์เพื่อหมุนเส้นสีแดงให้ทาบตรงกับร่องเกลียวสับปะรด"
            )
            st.session_state.manual_angle = current_theta

            # คำนวณ Brix จากมุมปัจจุบัน
            brix_val, diff_x, ideal_angle = calc_brix(current_theta, model_choice)

            st.markdown("---")
            st.subheader("📊 ผลการคำนวณ Brix")
            
            m1, m2 = st.columns(2)
            m1.metric("จำนวนตาที่พบ", f"{len(straight_eyes)} จุด")
            m2.metric("มุมของเส้นปัจจุบัน (θ)", f"{current_theta:.2f}°")

            st.metric("🍬 ค่าความหวานประเมิน", f"{brix_val:.2f} °Brix")
            st.info(f"📍 มุมอุดมคติ: `{ideal_angle:.1f}°` | ผลต่าง ($x$): `{diff_x:.2f}°`")

        with col1:
            # วาดเส้นหมุนวัดมุมบนภาพที่ดัดตั้งตรงเรียบร้อยแล้ว
            out_img = draw_hud_smooth(straight_cv_img, straight_eyes, current_theta)
            st.image(
                out_img, 
                caption=f"ภาพสับปะรดถูกดัดตั้งตรงแล้ว | กำลังวัดมุมเกลียวที่ {current_theta:.2f}°", 
                use_container_width=True
            )

    else:
        st.warning("⚠️ ตรวจจับจุดตาได้น้อยกว่า 2 จุด ไม่สามารถสร้างจุดหมุนวัดมุมได้")
        st.image(image, use_container_width=True)

    if os.path.exists(temp_path):
        os.remove(temp_path)
