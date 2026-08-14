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
st.title("🍍 ระบบปรับดัดรูป + หมุนเส้นวัดมุมเกลียว & Brix Calculator")

# -----------------------------------------------------------------------------
# 2. YOLO Model Load
# -----------------------------------------------------------------------------
@st.cache_resource
def load_yolo():
    return YOLO("best.pt")

# -----------------------------------------------------------------------------
# 3. Image Processing & Helper Functions
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

def auto_detect_tilt_angle(centroids):
    """คำนวณมุมเอียงอัตโนมัติของผลสับปะรดจากกลุ่มจุดตาด้วย Pure NumPy"""
    if len(centroids) < 2:
        return 0.0

    pts = np.array(centroids, dtype=np.float32)
    mean_x = float(np.mean(pts[:, 0]))
    mean_y = float(np.mean(pts[:, 1]))

    centered_pts = pts - np.array([mean_x, mean_y])
    cov_matrix = np.cov(centered_pts, rowvar=False)
    
    if cov_matrix.shape == (2, 2):
        evals, evecs = np.linalg.eigh(cov_matrix)
        primary_vec = evecs[:, np.argmax(evals)]
        angle_rad = math.atan2(primary_vec[1], primary_vec[0])
        tilt_deg = math.degrees(angle_rad)
    else:
        tilt_deg = 0.0

    if tilt_deg > 45:
        correction_deg = tilt_deg - 90
    elif tilt_deg < -45:
        correction_deg = tilt_deg + 90
    else:
        correction_deg = tilt_deg

    return correction_deg

def rotate_image_and_points(cv_img, centroids, angle_deg):
    """หมุนรูปภาพและคำนวณตำแหน่งพิกัดจุดตาใหม่ตามองศาที่กำหนด"""
    if len(centroids) == 0 or abs(angle_deg) < 0.01:
        return cv_img, centroids

    h, w = cv_img.shape[:2]
    pts = np.array(centroids, dtype=np.float32)
    mean_x = float(np.mean(pts[:, 0]))
    mean_y = float(np.mean(pts[:, 1]))

    # สร้าง Matrix สำหรับหมุนภาพรอบจุดศูนย์กลางกลุ่มตา
    rot_matrix = cv2.getRotationMatrix2D((mean_x, mean_y), angle_deg, 1.0)
    rotated_img = cv2.warpAffine(
        cv_img, rot_matrix, (w, h), 
        flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(255, 255, 255)
    )

    # Transform พิกัดจุดตาตามภาพที่หมุน
    ones = np.ones(shape=(len(centroids), 1))
    points_ones = np.hstack([pts, ones])
    transformed_pts = rot_matrix.dot(points_ones.T).T
    new_centroids = [(p[0], p[1]) for p in transformed_pts]

    return rotated_img, new_centroids

def calculate_auto_angle(centroids, img_h):
    """คำนวณมุมเส้นวัดเกลียว Auto ตั้งต้น"""
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

    # 1. พล็อตจุดตา
    for cx, cy in centroids:
        pt = (int(cx), int(cy))
        cv2.circle(img_out, pt, 6, (0, 255, 127), 2, cv2.LINE_AA)  # วงเขียว
        cv2.circle(img_out, pt, 2, (0, 165, 255), -1, cv2.LINE_AA) # จุดส้ม

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

        # 2. วาดเส้นขนานพื้น (Baseline 180° - สีฟ้า)
        cv2.line(img_out, (x1, int(mean_y)), (x2, int(mean_y)), (255, 200, 0), 2, cv2.LINE_AA)

        # 3. วาดเส้นหมุนวัดมุมตาม Slider (Measurement Line - สีแดง)
        cv2.line(img_out, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)

        # 4. วาดส่วนโค้งแสดงองศา (Angle Arc)
        cv2.ellipse(img_out, (int(mean_x), int(mean_y)), (45, 45), 0, 180, 180 - phi_deg, (0, 255, 255), 2, cv2.LINE_AA)

    return PIL.Image.fromarray(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))

def calc_brix(theta, model_name):
    """คำนวณ Brix จากมุมเส้นวัด"""
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
# 5. Main UI
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader("อัปโหลดรูปภาพสับปะรด", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1.2, 1])

    image = PIL.Image.open(uploaded_file).convert("RGB")
    temp_path = "temp_img.jpg"
    image.save(temp_path)
    img_w, img_h = image.size

    # 1. โหลด YOLO ตรวจจับจุดตา
    try:
        model = load_yolo()
        raw_eyes = get_filtered_eyes(temp_path, model, img_w, img_h, eye_dist_ratio)
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการรัน YOLO: {e}")
        raw_eyes = []

    if len(raw_eyes) >= 2:
        cv_img_orig = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # ---------------------------------------------------------------------
        # ส่วนควบคุมการหมุนรูปภาพ (Image Rotation Controls)
        # ---------------------------------------------------------------------
        with col2:
            st.subheader("🖼️ 1. ปรับการหมุนรูปภาพ (Image Rotation)")
            
            auto_align = st.checkbox("⚡ เปิดใช้งานระบบหมุนตั้งภาพ Auto", value=False, help="หากเปิด ระบบจะพยายามคำนวณและตั้งภาพให้อัตโนมัติ")

            auto_tilt = auto_detect_tilt_angle(raw_eyes) if auto_align else 0.0

            if "img_rotation" not in st.session_state:
                st.session_state.img_rotation = float(round(auto_tilt, 2))

            if auto_align:
                st.session_state.img_rotation = float(round(auto_tilt, 2))

            # สไลเดอร์หมุนรูปภาพด้วยตนเอง (Manual Image Rotate)
            img_angle = st.slider(
                "หมุนปรับระดับรูปภาพ (องศา):",
                min_value=-90.0,
                max_value=90.0,
                value=float(st.session_state.img_rotation),
                step=0.5,
                format="%.1f°",
                help="ลากปรับเพื่อตั้งภาพสับปะรดให้อยู่ในแนวตั้งตรงตามต้องการ",
                disabled=auto_align # ถ้าเปิด Auto จะล็อคสไลเดอร์ไว้
            )
            
            if not auto_align:
                st.session_state.img_rotation = img_angle

            if st.button("🔄 รีเซ็ตการหมุนรูปภาพเป็น 0°"):
                st.session_state.img_rotation = 0.0
                st.rerun()

            st.markdown("---")

            # -----------------------------------------------------------------
            # ส่วนหมุนเส้นวัดมุมเกลียวสับปะรด (Thread Line Measurement)
            # -----------------------------------------------------------------
            st.subheader("📐 2. หมุนเส้นวัดมุมเกลียว (Thread Line)")

            # คำนวณพิกัดตาหลังจากหมุนรูปภาพ
            rotated_cv_img, rotated_eyes = rotate_image_and_points(cv_img_orig, raw_eyes, st.session_state.img_rotation)
            auto_theta = calculate_auto_angle(rotated_eyes, img_h)

            if "manual_angle" not in st.session_state:
                st.session_state.manual_angle = float(round(auto_theta, 2))

            if st.button("🎯 ดึงมุม Auto เป็นค่าเริ่มต้นสำหรับเส้นสีแดง"):
                st.session_state.manual_angle = float(round(auto_theta, 2))
                st.rerun()

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

            # คำนวณ Brix
            brix_val, diff_x, ideal_angle = calc_brix(current_theta, model_choice)

            st.markdown("---")
            st.subheader("📊 ผลการคำนวณ Brix")
            
            m1, m2 = st.columns(2)
            m1.metric("จำนวนตาที่พบ", f"{len(rotated_eyes)} จุด")
            m2.metric("มุมของเส้นปัจจุบัน (θ)", f"{current_theta:.2f}°")

            st.metric("🍬 ค่าความหวานประเมิน", f"{brix_val:.2f} °Brix")
            st.info(f"📍 มุมอุดมคติ: `{ideal_angle:.1f}°` | ผลต่าง ($x$): `{diff_x:.2f}°`")

        # ---------------------------------------------------------------------
        # แสดงผลรูปภาพฝั่งซ้าย (col1)
        # ---------------------------------------------------------------------
        with col1:
            out_img = draw_hud_smooth(rotated_cv_img, rotated_eyes, current_theta)
            st.image(
                out_img, 
                caption=f"รูปถูกหมุนไป: {st.session_state.img_rotation:.1f}° | วัดมุมเกลียวที่: {current_theta:.2f}°", 
                use_container_width=True
            )

    else:
        st.warning("⚠️ ตรวจจับจุดตาได้น้อยกว่า 2 จุด ไม่สามารถหมุนรูปหรือวัดมุมได้")
        st.image(image, use_container_width=True)

    if os.path.exists(temp_path):
        os.remove(temp_path)
