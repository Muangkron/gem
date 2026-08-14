import math
import os
import cv2
import numpy as np
import PIL.Image
import PIL.ImageDraw
import streamlit as st
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# 1. Page Config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="🍍 Pineapple Brix Calculation System",
    page_icon="🍍",
    layout="wide",
)

st.title("🍍 ระบบคำนวณความหวานสับปะรด (Auto-Angle + Error Offset + Manual Model)")
st.caption("ระบบตรวจจับพิกัดตา คำนวณมุมอัตโนมัติ ปรับแก้ Error ได้ด้วย Slider และคำนวณ Brix แบบเรียลไทม์")

# -----------------------------------------------------------------------------
# 2. Sidebar Settings
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ ปรับแต่งระบบ")
    
    dist_threshold_ratio = st.slider(
        "ระยะกรองจุดตาซ้ำ (% ของภาพ):",
        min_value=1.0,
        max_value=6.0,
        value=2.5,
        step=0.5
    )
    
    st.markdown("---")
    st.header("🍍 เลือกโมเดลสับปะรด")
    selected_model_type = st.radio(
        "เลือกแบบจำลองสับปะรด:",
        options=[
            "Model 5-8-13 (ตาใหญ่ ร่องห่าง / มุมอุดมคติ 155°)",
            "Model 8-13-21 (ตาเล็ก ร่องถี่อัดแน่น / มุมอุดมคติ 136°)"
        ]
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
    """ใช้ YOLO ตรวจจับพิกัดตา และกรองจุดซ้ำ"""
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
    """คำนวณมุมเกลียวสับปะรด (theta) ด้วยคณิตศาสตร์ Vector Regression"""
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
    """วาดจุดพิกัดตาและเส้นเกลียวสับปะรดบนภาพ"""
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

def calculate_brix(theta_val, model_choice):
    """คำนวณค่า °Brix ด้วยสูตรคณิตศาสตร์ Local Python"""
    if "Model 5-8-13" in model_choice:
        ideal_angle = 155.0
        x = abs(theta_val - ideal_angle)
        brix = (-0.0196 * (x ** 2)) + (0.0045 * x) + 16.757
        formula_str = f"Brix = (-0.0196 × {x:.2f}²) + (0.0045 × {x:.2f}) + 16.757"
    else:
        ideal_angle = 136.0
        x = abs(theta_val - ideal_angle)
        brix = (0.0082 * (x ** 2)) - (0.6667 * x) + 16.362
        formula_str = f"Brix = (0.0082 × {x:.2f}²) - (0.6667 × {x:.2f}) + 16.362"
        
    return brix, x, ideal_angle, formula_str

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
    st.subheader("2. ประมวลผลและคำนวณ Brix")

    if uploaded_file is not None:
        # ตรวจจับพิกัดตาด้วย YOLO
        try:
            yolo_model = load_yolo_model("best.pt")
            centroids = detect_and_filter_eyes(temp_path, yolo_model, img_w, img_h, dist_threshold_ratio)
        except Exception as e:
            st.error("เกิดข้อผิดพลาดในการรันโมเดล YOLO:")
            st.exception(e)
            centroids = []

        if len(centroids) < 2:
            st.warning("⚠️ ตรวจจับตาได้น้อยกว่า 2 จุด ไม่สามารถคำนวณมุมเกลียวได้")
        else:
            # คำนวณมุมต้นฉบับจากภาพ
            slope, intercept, phi, raw_theta = calculate_accurate_spiral_angle(centroids, img_w, img_h)

            # แสดงภาพ Overlay
            overlay_img = draw_visual_overlay(image, centroids, slope, intercept, phi, raw_theta)
            st.image(overlay_img, caption=f"พิกัดตาและเส้นเวกเตอร์มุมเกลียว", use_container_width=True)

            st.markdown("---")
            st.subheader("🛠️ ปรับแต่งค่าองศา Error Offset")
            
            # Slider ปรับชดเชยค่า Error (+/- 15 องศา)
            angle_offset = st.slider(
                "ปรับชดเชยมุม Offset (°):",
                min_value=-15.0,
                max_value=15.0,
                value=0.0,
                step=0.1,
                help="ใช้ปรับบวก/ลบค่าองศาหากถ่ายภาพเอียงหรือมีค่า Error"
            )

            # คำนวณมุมสุดท้ายหลังปรับ Offset
            final_theta = raw_theta + angle_offset

            # แสดงผลการเปรียบเทียบองศา
            m1, m2 = st.columns(2)
            m1.metric("มุมที่วัดได้จริงจากภาพ (Raw θ)", f"{raw_theta:.2f}°")
            m2.metric("มุมสุทธิหลังปรับ Offset (Final θ)", f"{final_theta:.2f}°", delta=f"{angle_offset:+.1f}°")

            # คำนวณค่า Brix ทันที
            brix_val, x_diff, ideal_deg, formula_used = calculate_brix(final_theta, selected_model_type)

            st.markdown("---")
            st.markdown("### 📊 ผลการคำนวณค่าความหวาน")
            
            st.success(f"🍬 **ค่าความหวานประเมิน:** `{brix_val:.2f} °Brix`")
            
            with st.expander("🔍 ดูรายละเอียดการคำนวณ"):
                st.write(f"- **โมเดลที่เลือก:** {selected_model_type}")
                st.write(f"- **มุมอุดมคติของโมเดล:** {ideal_deg:.1f}°")
                st.write(f"- **ผลต่างองศา ($x = |\\theta - \\text{{ideal}}|$):** `{x_diff:.2f}°`")
                st.write(f"- **สมการที่ใช้:** `{formula_used}`")

        if os.path.exists(temp_path):
            os.remove(temp_path)
