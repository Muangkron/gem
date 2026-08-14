import math
import os
import cv2
import numpy as np
import PIL.Image
import streamlit as st
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="🍍 Precision Pineapple Brix Analyzer",
    page_icon="🍍",
    layout="wide",
)

st.title("🍍 ระบบประเมินความหวานสับปะรด (YOLO + OpenCV HUD + Error Offset)")
st.caption("วัดมุมเกลียวด้วยคณิตศาสตร์เวกเตอร์ แสดงผลกราฟิกชั้นสูงด้วย OpenCV และคำนวณ Brix ตามโมเดลที่เลือก")

# -----------------------------------------------------------------------------
# 2. Sidebar Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ การตั้งค่าระบบ")
    
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
        "ระบุประเภทโมเดล:",
        options=[
            "Model 5-8-13 (ตาใหญ่ ร่องห่าง / มุมอุดมคติ 155°)",
            "Model 8-13-21 (ตาเล็ก ร่องถี่อัดแน่น / มุมอุดมคติ 136°)"
        ]
    )

# -----------------------------------------------------------------------------
# 3. Load Trained YOLO Model
# -----------------------------------------------------------------------------
@st.cache_resource
def load_yolo_model(model_path="best.pt"):
    return YOLO(model_path)

# -----------------------------------------------------------------------------
# 4. Computer Vision & Math Functions
# -----------------------------------------------------------------------------
def detect_and_filter_eyes(image_path, model, img_w, img_h, ratio=2.5):
    """ใช้ YOLO ตรวจจับตำแหน่งตา และกรองจุดซ้ำด้วยคำนวณระยะทาง Euclidean"""
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
    """คำนวณความชันและมุมเกลียวสับปะรด (theta) ด้วยเวกเตอร์ Linear Regression"""
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

def draw_opencv_hud_overlay(pil_img, centroids, slope, intercept, theta_deg):
    """วาด Overlay ด้วย OpenCV สไตล์ HUD คมชัดและสวยงาม"""
    # แปลง PIL Image เป็น OpenCV BGR Format
    img_np = np.array(pil_img)
    cv_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    overlay = cv_img.copy()
    h, w, _ = cv_img.shape

    # 1. วาดจุดตำแหน่งตา (Eye Target Centroids)
    r_outer = max(6, int(min(w, h) * 0.012))
    r_inner = max(2, int(r_outer * 0.35))

    for cx, cy in centroids:
        pt = (int(cx), int(cy))
        # วงกลมชั้นนอก (สีเขียวสะท้อนแสง)
        cv2.circle(cv_img, pt, r_outer, (0, 255, 127), 2, lineType=cv2.LINE_AA)
        # จุดแกนกลาง (สีส้มสด)
        cv2.circle(cv_img, pt, r_inner, (0, 165, 255), -1, lineType=cv2.LINE_AA)

    if slope is not None and intercept is not None:
        mean_x = int(np.mean([p[0] for p in centroids]))
        mean_y = int(np.mean([p[1] for p in centroids]))
        line_len = int(w * 0.38)

        x1, x2 = max(0, mean_x - line_len), min(w, mean_x + line_len)
        y1, y2 = int(slope * x1 + intercept), int(slope * x2 + intercept)

        # 2. วาดเส้นอ้างอิงแนวนอน (Horizontal Reference Baseline - สีฟ้า)
        cv2.line(cv_img, (x1, mean_y), (x2, mean_y), (255, 200, 0), 2, cv2.LINE_AA)

        # 3. วาดเส้นเกลียวสับปะรด (Spiral Tangent Vector Line - สีแดงนีออน)
        cv2.line(cv_img, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)

        # 4. วาดกล่อง HUD สรุปข้อมูลแบบกึ่งโปร่งแสง (Semi-transparent HUD Badge)
        badge_w, badge_h = int(w * 0.45), 70
        bx1, by1 = 20, 20
        bx2, by2 = bx1 + badge_w, by1 + badge_h

        # สร้างพื้นหลังดำโปร่งแสง 60%
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.6, cv_img, 0.4, 0, cv_img)
        cv2.rectangle(cv_img, (bx1, by1), (bx2, by2), (0, 255, 255), 2, lineType=cv2.LINE_AA)

        # พิมพ์ข้อความแสดงค่าองศา
        text_title = f"DETECTED EYES: {len(centroids)} PTS"
        text_angle = f"SPIRAL ANGLE (Theta): {theta_deg:.2f} deg"
        
        cv2.putText(cv_img, text_title, (bx1 + 15, by1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(cv_img, text_angle, (bx1 + 15, by1 + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    # แปลง OpenCV BGR กลับเป็น RGB ส่งออกให้ Streamlit แสดงผล
    rgb_result = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return PIL.Image.fromarray(rgb_result)

def calculate_brix(theta_val, model_choice):
    """คำนวณค่า °Brix ด้วยสูตรคณิตศาสตร์ Local"""
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
# 5. Main Application UI Layout
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
    st.subheader("2. ผลการประมวลผล OpenCV & Brix")

    if uploaded_file is not None:
        try:
            yolo_model = load_yolo_model("best.pt")
            centroids = detect_and_filter_eyes(temp_path, yolo_model, img_w, img_h, dist_threshold_ratio)
        except Exception as e:
            st.error("เกิดข้อผิดพลาดในการโหลดหรือรันโมเดล YOLO (best.pt):")
            st.exception(e)
            centroids = []

        if len(centroids) < 2:
            st.warning("⚠️ ตรวจจับตาได้น้อยกว่า 2 จุด ไม่สามารถสร้างเส้นเวกเตอร์คำนวณมุมได้")
        else:
            # คำนวณมุมเกลียว
            slope, intercept, phi, raw_theta = calculate_accurate_spiral_angle(centroids, img_w, img_h)

            # วาดผลลัพธ์ด้วย OpenCV
            hud_image = draw_opencv_hud_overlay(image, centroids, slope, intercept, raw_theta)
            st.image(hud_image, caption="ผลการประมวลผลพิกัดตาและมุมเกลียว (OpenCV HUD)", use_container_width=True)

            st.markdown("---")
            st.subheader("🛠️ ปรับแต่งค่าองศา Error Offset")
            
            angle_offset = st.slider(
                "ปรับชดเชยมุม Offset (°):",
                min_value=-15.0,
                max_value=15.0,
                value=0.0,
                step=0.1,
                help="ใช้ปรับบวก/ลบค่าองศาเพื่อชดเชยุมุมเอียงจากการถ่ายภาพ"
            )

            # มุมสุทธิหลังปรับ Slider
            final_theta = raw_theta + angle_offset

            m1, m2 = st.columns(2)
            m1.metric("มุมที่วัดได้จริงจากภาพ (Raw θ)", f"{raw_theta:.2f}°")
            m2.metric("มุมสุทธิหลังปรับ Offset (Final θ)", f"{final_theta:.2f}°", delta=f"{angle_offset:+.1f}°")

            # คำนวณ Brix
            brix_val, x_diff, ideal_deg, formula_used = calculate_brix(final_theta, selected_model_type)

            st.markdown("---")
            st.markdown("### 📊 ผลการประเมินความหวาน")
            
            st.success(f"🍬 **ค่าความหวานประเมิน:** `{brix_val:.2f} °Brix`")
            
            with st.expander("🔍 ดูรายละเอียดขั้นตอนการคำนวณ"):
                st.write(f"- **โมเดลที่เลือก:** {selected_model_type}")
                st.write(f"- **มุมอุดมคติของโมเดล:** {ideal_deg:.1f}°")
                st.write(f"- **ผลต่างองศา ($x = |\\theta - \\text{{ideal}}|$):** `{x_diff:.2f}°`")
                st.write(f"- **สมการที่ใช้คำนวณ:** `{formula_used}`")

        if os.path.exists(temp_path):
            os.remove(temp_path)
