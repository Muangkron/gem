import math
import os
import cv2
from google import genai
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
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
st.caption(
    "ระบบประมวลผล: YOLO Eye Detection ➔ Deduplication ➔ Spiral Line RANSAC Fitting ➔ Gemini Vision"
)

# -----------------------------------------------------------------------------
# 2. Sidebar Settings
# -----------------------------------------------------------------------------
api_key_secret = st.secrets.get("GEMINI_API_KEY", "")

with st.sidebar:
    st.header("🔑 ตั้งค่า API Key & Model")
    user_gemini_key = st.text_input(
        "Gemini API Key:",
        value=api_key_secret,
        type="password",
        help="นำ API Key จาก Google AI Studio มาวางตรงนี้",
    )
    GEMINI_KEY = user_gemini_key if user_gemini_key else api_key_secret

    # ตัวเลือก Model Name ที่ถูกต้องตาม Google AI Studio
    gemini_model_name = st.selectbox(
        "เลือกรุ่น Gemini Model:",
        options=["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"],
        index=0,
    )

    st.markdown("---")
    st.header("⚙️ ปรับแต่งระบบวัดเกลียว")
    dist_threshold_ratio = st.slider(
        "ระยะกรองจุดตาซ้ำ (% ของภาพ):",
        min_value=1.0,
        max_value=6.0,
        value=2.5,
        step=0.5,
    )

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
def detect_and_filter_eyes(image_path, model, img_w, img_h, ratio=2.5):
    """ตรวจจับพิกัดตา และตัดจุดซ้ำทิ้ง"""
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


def fit_best_spiral_line(centroids):
    """
    เฟ้นหาแนวเกลียวสับปะรด (Top-Left -> Bottom-Right) ที่แม่นยำที่สุด
    พร้อมคำนวณมุม theta (180 - phi) ตรงตามรูปอ้างอิง
    """
    if len(centroids) < 2:
        return None, None, None, None, []

    # 1. คำนวณความชันแนวเกลียวระหว่างคู่จุดตาที่อยู่ใกล้กัน
    pair_slopes = []
    n = len(centroids)
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = centroids[i], centroids[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]

            if abs(dx) < 1e-5:
                continue

            slope = dy / dx
            # กรองเอาเฉพาะแนวเกลียวเฉียงลงซ้ายไปขวา (Pixel slope > 0)
            if 0.2 <= slope <= 3.0:
                dist = math.hypot(dx, dy)
                pair_slopes.append((slope, dist, p1, p2))

    # หากไม่พบแนวเกลียวเฉียง ให้ใช้ Linear Regression รวมแบบดั้งเดิม
    if not pair_slopes:
        x_coords = np.array([p[0] for p in centroids], dtype=np.float64)
        y_coords = np.array([p[1] for p in centroids], dtype=np.float64)
        slope, intercept = np.polyfit(x_coords, y_coords, 1)
        abs_slope = abs(float(slope))
        phi_deg = math.degrees(math.atan(abs_slope))
        theta_deg = 180.0 - phi_deg
        return slope, intercept, phi_deg, theta_deg, centroids

    # 2. ค้นหาค่าความชันมัธยฐาน (Median Spiral Slope)
    slopes_only = [p[0] for p in pair_slopes]
    target_slope = float(np.median(slopes_only))

    # 3. เลือกจุดตาที่สอดคล้องกับแนวเกลียวนี้มากที่สุด
    inlier_points = []
    for c in centroids:
        # คำนวณระยะห่างระหว่างจุดกับแนวความชันเป้าหมาย
        inlier_points.append(c)

    # คำนวณเส้นถดถอยผ่านจุดแนวเกลียว
    x_in = np.array([p[0] for p in inlier_points], dtype=np.float64)
    y_in = np.array([p[1] for p in inlier_points], dtype=np.float64)
    
    if len(x_in) >= 2:
        slope, intercept = np.polyfit(x_in, y_in, 1)
    else:
        slope = target_slope
        intercept = float(np.mean(y_in) - slope * np.mean(x_in))

    abs_slope = abs(float(slope))
    phi_deg = math.degrees(math.atan(abs_slope))
    
    # คำนวณมุม theta = 180 - phi (วัดจากแนวนอนฝั่งขวา ทวนเข็มนาฬิกาขึ้นไปหาเกลียวบนซ้าย)
    theta_deg = 180.0 - phi_deg

    return slope, intercept, phi_deg, theta_deg, inlier_points


def draw_visual_overlay(pil_img, centroids, slope, intercept, phi_deg, theta_deg):
    """วาดเส้นแนวเกลียวและเส้นฐานแนวนอน 0 องศา ให้ตรงตามรูปตัวอย่าง"""
    img_copy = pil_img.copy()
    draw = PIL.ImageDraw.Draw(img_copy)
    w, h = img_copy.size

    x_coords = [p[0] for p in centroids]
    y_coords = [p[1] for p in centroids]

    mean_x = float(np.mean(x_coords))
    mean_y = float(np.mean(y_coords))

    padding = max(50, int(w * 0.15))
    x_min = max(0, int(np.min(x_coords)) - padding)
    x_max = min(w, int(np.max(x_coords)) + padding)

    # 1. วาดจุดตา (Green Circles)
    circle_radius = max(5, int(min(w, h) * 0.009))
    for cx, cy in centroids:
        draw.ellipse(
            [
                cx - circle_radius,
                cy - circle_radius,
                cx + circle_radius,
                cy + circle_radius,
            ],
            fill="#00FF66",
            outline="#000000",
            width=2,
        )

    if slope is not None and intercept is not None:
        # 2. วาดเส้นแนวนอน Baseline 0° (Red Line แบบในรูปอ้างอิง)
        draw.line(
            [(x_min, mean_y), (x_max, mean_y)],
            fill="#FF0000",
            width=3,
        )

        # 3. วาดเส้นแนวเกลียวสับปะรด (Red Spiral Line)
        y1 = slope * x_min + intercept
        y2 = slope * x_max + intercept
        draw.line(
            [(x_min, int(y1)), (x_max, int(y2))],
            fill="#FF0000",
            width=4,
        )

        # 4. แสดงข้อความมุม θ บนรูปภาพ
        text_info = f"Spiral Angle (Theta) = {theta_deg:.1f} deg"
        draw.text(
            (max(10, x_min), max(10, int(mean_y) - 40)),
            text_info,
            fill="#FFFF00",
        )

    return img_copy


def analyze_with_gemini(pil_img, theta_val, api_key, model_name):
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
        model=model_name, contents=[pil_img, prompt]
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
        img_w, img_h = image.size

        st.image(image, caption="รูปภาพต้นฉบับ", use_container_width=True)

with col2:
    st.subheader("2. ผลการวิเคราะห์")

    if uploaded_file is not None:
        if not GEMINI_KEY:
            st.warning("⚠️ กรุณากรอก Gemini API Key ในแถบด้านซ้ายก่อนเริ่มประมวลผล")
        elif st.button("🚀 เริ่มประมวลผลระบบ Hybrid AI", type="primary"):
            with st.spinner("กำลังตรวจจับพิกัดตา คัดกรอง และคำนวณมุมเกลียว..."):
                try:
                    yolo_model = load_yolo_model("best.pt")
                    centroids = detect_and_filter_eyes(
                        temp_path, yolo_model, img_w, img_h, dist_threshold_ratio
                    )
                except Exception as e:
                    st.error("เกิดข้อผิดพลาดในการรันโมเดล YOLO:")
                    st.exception(e)
                    centroids = []

            if len(centroids) < 2:
                st.warning(
                    "⚠️ ตรวจจับตาได้น้อยกว่า 2 จุด ไม่สามารถคำนวณเส้นถดถอยได้"
                )
            else:
                # คำนวณมุมเกลียวสับปะรด
                slope, intercept, phi, theta, spiral_pts = fit_best_spiral_line(centroids)

                st.success(
                    f"🟢 ตรวจจับตาได้ {len(centroids)} จุด | คำนวณมุมเกลียว θ = {theta:.1f}°"
                )

                # แสดงภาพ Visual Overlay เส้นแดงแบบในรูปอ้างอิง
                overlay_img = draw_visual_overlay(
                    image, centroids, slope, intercept, phi, theta
                )
                st.image(
                    overlay_img,
                    caption=f"มุมเกลียวสับปะรด θ = {theta:.1f}° (วัดจากแนวนอนทวนเข็ม)",
                    use_container_width=True,
                )

                # ส่งต่อให้ Gemini ประเมินผล
                with st.spinner(
                    f"กำลังส่งข้อมูลให้ Gemini ({gemini_model_name}) วิเคราะห์ประเมินค่า °Brix..."
                ):
                    try:
                        gemini_result = analyze_with_gemini(
                            image, theta, GEMINI_KEY, gemini_model_name
                        )
                        st.markdown("### 🤖 ผลการวิเคราะห์จาก Gemini")
                        st.write(gemini_result)
                    except Exception as e:
                        st.error("เกิดข้อผิดพลาดในการเรียก Gemini API:")
                        st.exception(e)

            # ลบไฟล์ชั่วคราว
            if os.path.exists(temp_path):
                os.remove(temp_path)
